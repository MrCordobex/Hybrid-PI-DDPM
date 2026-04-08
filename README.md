# TFM Shell Research Project

Proyecto de investigacion para generacion condicional de cubiertas laminares con alto
factor membranar (`mf`) mediante un esquema de dos modelos:

1. Un **architect** difusivo que aprende la distribucion geometrica de las superficies.
2. Un **engineer** tipo **Physics-Based Parallel UNet** que aproxima la respuesta
   mecanica y aporta la guia fisica durante la generacion.

El objetivo cientifico es generar una geometria `z` coherente con una carga vertical
`fz`, sin filtrar el dataset por `mf`, y despues sesgar la sintesis hacia soluciones
con comportamiento lo mas membranar posible.

## 1. Idea cientifica del proyecto

La tesis que implementa este repositorio es la siguiente:

- El **architect** debe aprender la variedad de geometria estructural usando todas las
  muestras disponibles, sin introducir fisica en la funcion de perdida.
- El **engineer** debe aprender el operador mecanico
  `(z, fz) -> (uz, membrana, flexion)` con una arquitectura paralela alineada con la
  formulacion fisica del problema.
- La **generacion final** se realiza con difusion guiada: el architect propone y el
  engineer corrige mediante gradientes de una magnitud objetivo basada en `mf`.

En lenguaje probabilistico:

- El architect modela un prior aproximado `p_theta(z | fz)`.
- El engineer actua como un surrogate diferenciable de la fisica
  `g_phi(z_t, fz, t) ~= y`.
- La generacion guiada combina ambos para favorecer muestras con `mf` alto.

## 2. Estructura del proyecto

```text
TFM/
  artifacts/                # Salidas de cada run: curvas, figuras, summaries
  configs/                  # Configs YAML de entrenamiento y muestreo
  data/                     # Carpeta canonica para datos auxiliares del proyecto
  main.py                   # CLI unificada
  mlruns/                   # Tracking local de MLflow
  models/                   # Checkpoints, stats, splits e historicos
  pyproject.toml            # Proyecto Python gestionado con uv
  src/tfm_shells/
    cli.py
    config.py
    constants.py
    data/
      dataset.py
      index.py
    models/
      factory.py
      parallel_pb_unet.py
    sampling/
      guided.py
    training/
      common.py
      train_architect.py
      train_engineer.py
    utils/
      io.py
      physics.py
      tracking.py
  train_architect.py
  train_engineer.py
  sample_guided.py
```

## 3. Dataset y notacion

### 3.1 Archivos esperados

El codigo trabaja con ficheros `.npz` del tipo:

```text
shell_1200.npz
shell_2600.npz
shell_hole_999.npz
```

Cada archivo contiene, como minimo, los campos:

```text
z, fz, mf, ds, dv, uz,
se11, se22, se12,
sf11, sf22, sf12,
sk11, sk22, sk12,
sm11, sm22, sm12
```

### 3.2 Malla y variables

Cada muestra vive sobre una malla fija `64 x 64`. En la implementacion actual:

- `z in R^(1 x 64 x 64)` es la geometria escalar normalizada despues a `[-1, 1]`.
- `fz in R^(1 x 64 x 64)` es el campo de carga vertical.
- `uz in R^(1 x 64 x 64)` es el desplazamiento vertical.
- `seij` representa deformaciones membranares.
- `sfij` representa esfuerzos/resultantes de membrana.
- `skij` representa curvaturas de flexion.
- `smij` representa momentos/resultantes de flexion.
- `ds` es el elemento de superficie discreto.
- `dv` es el elemento de volumen discreto.
- `mf in [0, 1]^(1 x 64 x 64)` es el factor membranar por pixel.

La organizacion de canales fisicos usada por el repositorio es:

```text
y =
[
  uz,
  se11, se22, se12, sf11, sf22, sf12,
  sk11, sk22, sk12, sm11, sm22, sm12
]
in R^(13 x 64 x 64)
```

Por tanto:

- Rama `u`: 1 canal.
- Rama `m`: 6 canales.
- Rama `f`: 6 canales.

### 3.3 Normalizacion

La geometria y la carga se normalizan con min-max global de entrenamiento:

```text
z_norm = 2 * (z - z_min) / (z_max - z_min) - 1
fz_norm = 2 * (fz - fz_min) / (fz_max - fz_min) - 1
```

Los campos fisicos del engineer se estandarizan canal a canal:

```text
y_norm[c] = (y[c] - mu_c) / sigma_c
```

donde `mu_c` y `sigma_c` se calculan solo con el split de entrenamiento.

## 4. Formulacion matematica del architect

### 4.1 Rol del architect

El architect no usa fisica en la loss. Su trabajo es aprender la distribucion de
geometrias `z` condicionadas por `fz`.

En la implementacion:

- Usa **todas** las estructuras por defecto (`subset: all`).
- No aplica ningun filtrado por `mf`.
- Usa `1000` timesteps de difusion.
- Trabaja con `z_norm` y `fz_norm`.

### 4.2 Proceso forward

Sea `x_0 = z_norm`. El proceso de difusion directa se define como:

```text
q(x_t | x_0) = N(x_t ; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

Equivalentemente:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,
eps ~ N(0, I)
```

### 4.3 Proceso reverse condicionado

El modelo aprende un campo de denoising condicionado por la carga:

```text
eps_theta = eps_theta(x_t, fz_norm, t)
```

o, en este caso concreto, una parametrizacion tipo `v_prediction`:

```text
v_theta = v_theta(x_t, fz_norm, t)
```

La red recibe como entrada:

```text
input_architect = concat(x_t, fz_norm)
```

con `in_channels = 2`.

### 4.4 Funcion de perdida del architect

La loss implementada es una MSE pura contra el target asociado al scheduler:

```text
L_arch(theta) = E[ || target_t - model_theta(x_t, fz_norm, t) ||_2^2 ]
```

Si `prediction_type = v_prediction`, el target es:

```text
target_t = v(x_0, eps, t)
```

No aparece ningun termino de fisica en esta etapa.

## 5. Formulacion matematica del engineer

### 5.1 Rol del engineer

El engineer es el surrogate fisico diferenciable del proyecto. Debe predecir la
respuesta mecanica limpia a partir de una geometria ruidosa y la carga vertical:

```text
g_phi : (x_t, fz_norm, t) -> y_norm
```

donde `y_norm` contiene 13 canales fisicos limpios.

### 5.2 Arquitectura paralela PB-UNet

La implementacion sigue la idea de `mortech.pdf`: tres ramas independientes, cada una
especializada en una familia fisica distinta:

```text
UNet_u(x_t, fz_norm, t) -> uz
UNet_m(x_t, fz_norm, t) -> [se11, se22, se12, sf11, sf22, sf12]
UNet_f(x_t, fz_norm, t) -> [sk11, sk22, sk12, sm11, sm22, sm12]
```

La salida total es la concatenacion:

```text
y_hat = concat(y_hat_u, y_hat_m, y_hat_f)
```

Cada rama es un `UNet2DModel` completo y todas comparten la misma entrada:

```text
input_engineer = concat(x_t, fz_norm)
```

### 5.3 Loss supervisada

La perdida supervisada principal es:

```text
L_sup(phi) = E[ || y_norm - y_hat_norm ||_2^2 ]
```

Ademas, el entrenamiento registra metricas separadas por ramas:

```text
L_u = MSE(uz_hat, uz)
L_m = MSE(membrana_hat, membrana)
L_f = MSE(flexion_hat, flexion)
```

Esto no cambia la loss total por defecto, pero si permite interpretar el aprendizaje
de cada subsistema fisico.

### 5.4 Energia de membrana, energia de flexion y trabajo externo

Una vez desnormalizada la prediccion:

```text
y_hat_real = y_hat_norm * sigma + mu
```

se calculan las densidades energeticas discretas:

```text
w_memb = (sf11 * se11 + sf22 * se22 + 2 * sf12 * se12) * ds
w_flex = (sm11 * sk11 + sm22 * sk22 + 2 * sm12 * sk12) * ds
w_ext  = fz * uz * dv
```

donde:

- `w_memb` representa energia interna de membrana.
- `w_flex` representa energia interna de flexion.
- `w_ext` representa trabajo externo.

### 5.5 Residuo fisico

El residuo energetico global por muestra es:

```text
Delta_P = sum(w_memb) + sum(w_flex) - sum(w_ext)
```

La penalizacion fisica implementada es:

```text
L_phys(phi) = (Delta_P)^2
```

### 5.6 Ponderacion temporal de la fisica

El engineer se entrena sobre `x_t`, no sobre `x_0`, por lo que el termino fisico se
pondera tambien con el timestep:

```text
omega(t) = (1 - t / T)^p
```

donde:

- `T = 1000` es el numero total de timesteps.
- `p = timestep_power`.

Ademas, la contribucion fisica crece por warmup en funcion de la epoca:

```text
lambda_epoch = 0, si epoch < warmup
lambda_epoch = lambda_max * progress(epoch), en otro caso
```

### 5.7 Loss total del engineer

La funcion de perdida entrenada es:

```text
L_eng(phi) = L_sup(phi) + lambda_epoch * E[ omega(t) * L_phys(phi) ]
```

Esta es la pieza mas importante del proyecto desde el punto de vista cientifico:

- obliga a que la prediccion no sea solo numericamente parecida al FEM,
- sino tambien coherente con el equilibrio energetico del sistema.

## 6. Factor membranar

El factor membranar por pixel se reconstruye a partir de la energia relativa de
membrana y flexion:

```text
mf = w_memb_local / (w_memb_local + w_flex_local + eps)
```

con

```text
w_memb_local = sf11 * se11 + sf22 * se22 + 2 * sf12 * se12
w_flex_local = sm11 * sk11 + sm22 * sk22 + 2 * sm12 * sk12
```

e `eps` pequeno para evitar division por cero.

Interpretacion:

- `mf ~= 1` significa comportamiento mayoritariamente membranar.
- `mf ~= 0` significa comportamiento mayoritariamente flexional.

## 7. Generacion guiada por fisica

### 7.1 Esquema general

La generacion final usa el architect como motor probabilistico y el engineer como
corrector fisico differentiable.

Dado un campo `fz` de condicion:

1. Se inicializa `x_T ~ N(0, I)`.
2. El architect propone el paso reverse.
3. El engineer evalua la calidad fisica de la muestra intermedia.
4. El gradiente de la calidad fisica modifica el update reverse.

### 7.2 Objetivo de guiado

Sea `mf_hat(x_t, fz, t)` el `mf` predicho por el engineer. El objetivo implementado es:

```text
J(x_t) = mean( (1 - mean(mf_hat))^2 )
```

En el codigo se usa:

```text
objective = ((1 - mf_mean)^2).mean()
```

Minimizar `J` equivale a empujar `mf_mean` hacia `1`.

### 7.3 Gradiente de guiado

Se calcula:

```text
g_t = grad_{x_t} J(x_t)
```

y se introduce en la salida del architect con un peso programado:

```text
model_output_guided = model_output_architect + sqrt(1 - alpha_bar_t) * s_t * g_t
```

donde `s_t` depende de:

- `guidance_scale`
- `guide_w_min`
- `guide_w_max`
- `guide_power`
- o la campana gaussiana definida por `bell_peak` y `bell_width`

Despues el scheduler calcula:

```text
x_{t-1} = Step(model_output_guided, t, x_t)
```

### 7.4 Interpretacion

El architect garantiza diversidad y plausibilidad geometrica.

El engineer inclina esa diversidad hacia una region del espacio de diseno donde la
respuesta estructural sea mas membranar.

## 8. Que hace exactamente cada script

### `train_architect.py`

- Carga config.
- Indexa todo el dataset.
- Filtra por `subset` y `min_mf_mean`.
- Divide en train/val con estratificacion por tipo cuando hay mezclas.
- Calcula estadisticas globales de normalizacion.
- Entrena el DDPM condicionado por `fz`.
- Guarda checkpoints `best.pt` y `last.pt`.
- Registra metricas y artefactos en MLflow.
- Genera muestras de validacion periodicas.

### `train_engineer.py`

- Carga config.
- Construye el dataset fisico de 13 canales.
- Entrena la PB-UNet paralela con tres ramas.
- Calcula `L_sup` y `L_phys`.
- Registra metricas globales y por ramas.
- Guarda curvas, validacion visual, checkpoints y resumen del run.

### `sample_guided.py`

- Carga un checkpoint de architect y otro de engineer.
- Lee un archivo fuente para obtener el `fz` de condicion.
- Normaliza `fz` con las estadisticas de cada modelo.
- Ejecuta la cadena reverse del architect.
- Corrige cada paso con gradientes del engineer.
- Exporta las geometria generadas y sus figuras.

### `main.py`

Expone una CLI unica:

```bash
uv run python main.py architect --config configs/architect.yaml
uv run python main.py engineer --config configs/engineer.yaml
uv run python main.py sample --config configs/sample_guided.yaml
```

## 9. MLflow y trazabilidad cientifica

Cada run registra en MLflow:

- hiperparametros completos de la config,
- tamano del dataset filtrado,
- tamano de train y val,
- estadisticas de normalizacion,
- curvas de entrenamiento,
- checkpoints `best` y `last`,
- figuras de muestras o validacion,
- `summary.json`,
- historico por epoca.

Esto permite responder preguntas de tesis como:

- que arquitectura generaliza mejor,
- cuando la penalizacion fisica mejora o perjudica,
- como evoluciona `mf_mae`,
- si el sesgo fisico empeora la diversidad geometrica,
- que combinacion de `guidance_scale` y schedule produce mejores cubiertas.

## 10. Configuracion por defecto actual

### Architect

- `subset: all`
- `min_mf_mean: null`
- `include_fz_channel: true`
- `in_channels: 2`
- `out_channels: 1`
- `prediction_type: v_prediction`
- `beta_schedule: squaredcos_cap_v2`
- `sample_inference_steps: 1000`

### Engineer

- `subset: all`
- `min_mf_mean: null`
- `kind: parallel_pb_unet`
- `in_channels: 2`
- `out_channels: 13`
- `branch_channels = {u: 1, m: 6, f: 6}`
- `lambda_max: 1e-3`
- `warmup_epochs: 10`

### Sampler guiado

- condicionamiento por `fz` tomado de un `.npz` real,
- `num_inference_steps: 1000`,
- schedule de guiado tipo `bell`,
- pesos y clipping configurables por YAML.

## 11. Comandos recomendados

### Instalar entorno

```bash
uv sync
```

### Entrenar architect

```bash
uv run python train_architect.py --config configs/architect.yaml
```

### Entrenar engineer

```bash
uv run python train_engineer.py --config configs/engineer.yaml
```

### Muestrear con guiado fisico

```bash
uv run python sample_guided.py --config configs/sample_guided.yaml
```

## 12. Estado metodologico del repositorio

El repositorio implementa ya el nucleo metodologico que necesitas:

- entrenamiento separado de architect y engineer,
- uso de todas las estructuras por defecto,
- inclusion explicita de `fz` en architect, engineer y sampler,
- surrogate fisico con tres ramas,
- residual energetico differentiable,
- guiado de difusion hacia `mf` alto,
- trazabilidad completa de experimentos.

En resumen, la lectura matematica del pipeline es:

```text
fz --> architect --> geometria candidata z
               \\-> engineer --> campos fisicos --> mf --> gradiente --> guia reverse
```

o, de forma mas formal:

```text
z* = arg sample from p_theta(z | fz) under guidance induced by g_phi(z_t, fz, t)
```

con el sesgo de optimizacion dirigido a maximizar la membranariedad de la respuesta.
