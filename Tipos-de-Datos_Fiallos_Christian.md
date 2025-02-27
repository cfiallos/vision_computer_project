---
marp: true
theme: default
class: lead
---

# Tipos de Datos en Python, NumPy y Pandas

---

# Introducción

- Exploraremos los tipos de datos en Python, NumPy y Pandas.
- Comprender estos tipos es fundamental para el manejo eficiente de datos.

---

# Tipos de Datos en Python

##  Numéricos
### Enteros
- **`int`**: Números enteros.
### Flotantes
- **`float`**: Números de punto flotante.
### Complejos
- **`complex`**: Números complejos.
---
### Ejemplo de Datos Numericos
```python
# Ejemplo de Entero
edad = 25
print(edad)  # Salida: 25

# Ejemplo de Flotante
altura = 1.75
print(altura)  # Salida: 1.75

# Ejemplo de Número Complejo
numero_complejo = 2 + 3j
print(numero_complejo)  # Salida: (2+3j)
```
---
## Secuencias
- **`str`**: Cadenas de texto.
- **`list`**: Listas mutables.
- **`tuple`**: Tuplas inmutables.
---
### Ejemplo de Secuencias

```python
# Ejemplo de Cadena de Texto
nombre = "Ana"
print(nombre)  # Salida: Ana

# Ejemplo de Lista
numeros = [1, 2, 3, 4, 5]
print(numeros)  # Salida: [1, 2, 3, 4, 5]

# Ejemplo de Tupla
coordenadas = (10, 20)
print(coordenadas)  # Salida: (10, 20)
```
---
## Conjuntos
- **`set`**: Conjuntos únicos.
- **`frozenset`**: Conjuntos inmutables.

### Ejemplo de Conjuntos

```python
# Ejemplo de Conjunto
frutas = {"manzana", "banana", "naranja"}
print(frutas)  # Salida: {'banana', 'naranja', 'manzana'}

# Ejemplo de Frozenset
frutas_inmutables = frozenset(["manzana", "banana", "naranja"])
print(frutas_inmutables)  # Salida: frozenset({'banana', 'naranja', 'manzana'})
```
---
## Diccionario

- **`dict`**: Diccionarios (pares clave-valor).

### Ejemplo de Diccionario

```python
persona = {"nombre": "Ana", "edad": 25}
print(persona)  # Salida: {'nombre': 'Ana', 'edad': 25}
```
---
## Booleanos
- **`bool`**: Representa valores de `True` o `False`.

### Ejemplo de Booleanos
```python
es_mayor_de_edad = True
print(es_mayor_de_edad)  # Salida: True
```
---
## NoneType

- **`None`**: Ausencia de valor o un valor nulo 

### Ejemplo de NoneType
```python
valor_nulo = None
print(valor_nulo)  # Salida: None
```
---

### Ejemplo de Tipos de Datos en Python
```python
# Ejemplo de varios tipos de datos
numero_entero = 10
numero_flotante = 3.14
numero_complejo = 2 + 3j
cadena = "Hola, mundo!"
mi_lista = [1, 2, 3]
mi_tupla = (10, 20)
mi_conjunto = {"manzana", "banana", "naranja"}
mi_conjunto_inmutable = frozenset(["manzana", "banana", "naranja"])
mi_diccionario = {'nombre': 'Juan', 'edad': 30}
es_mayor_de_edad = True
valor_nulo = None

print(numero_entero, numero_flotante, numero_complejo, cadena, mi_lista, mi_tupla, mi_conjunto, mi_conjunto_inmutable, mi_diccionario, es_mayor_de_edad, valor_nulo)
# Salida: 10 3.14 (2+3j) Hola, mundo! [1, 2, 3] (10, 20) {'banana', 'naranja', 'manzana'} frozenset({'banana', 'naranja', 'manzana'}) {'nombre': 'Juan', 'edad': 30} True None
```
---
# Tipos de Datos en NumPy
## Numéricos
### Enteros
- **`np.int8`, `np.int16`, `np.int32`, `np.int64`**: Enteros de 8, 16, 32 y 64 bits.
### Enteros sin signo
- **`np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`**: Enteros sin signo de 8, 16, 32 y 64 bits.
### Flotantes
- **`np.float16`, `np.float32`, `np.float64`**: Números de punto flotante de 16, 32 y 64 bits.
### Complejos
- **`np.complex64`, `np.complex128`**: Números complejos de 64 y 128 bits.
---
## Ejemplo de Datos Numéricos

```python
import numpy as np

# Ejemplo de Enteros
enteros = np.array([1, 2, 3], dtype=np.int32)
print(enteros)
# Salida: [1 2 3]

# Ejemplo de Enteros sin signo
enteros_sin_signo = np.array([1, 2, 3], dtype=np.uint32)
print(enteros_sin_signo)
# Salida: [1 2 3]

# Ejemplo de Flotantes
flotantes = np.array([1.1, 2.2, 3.3], dtype=np.float32)
print(flotantes)
# Salida: [1.1 2.2 3.3]

# Ejemplo de Números Complejos
complejos = np.array([1+2j, 3+4j], dtype=np.complex64)
print(complejos)
# Salida: [1.+2.j 3.+4.j]
```
---
## Booleanos
- **`bool`** : Representa valores de `True` o `False`.

## Ejemplo de Booleanos
```python
import numpy as np

booleanos = np.array([True, False, True], dtype=bool)
print(booleanos)
# Salida: [ True False  True]
```
---
## Cadenas de Caracteres

- **`string_`**: Cadena que tiene una longitud fija.


## Ejemplo de Cadenas de Caracteres
```python
import numpy as np

# Ejemplo de Cadenas de Caracteres
cadenas = np.array(["Hola", "Mundo"], dtype=np.string_)
print(cadenas)
# Salida: [b'Hola' b'Mundo']

```
---
## Objetos

- **`object_`**: contiene cualquier objeto de python.

## Ejemplo de Objetos
```python
import numpy as np

objetos = np.array([1, "Hola", 3.14], dtype=object)
print(objetos)
# Salida: [1 'Hola' 3.14]
```
---
## Datos Estructurados

- **`void`**: tipo de dato sin tipo específico.

## Ejemplo de Datos Estructurados
```python
import numpy as np

# Definir datos estructurados con un campo de tipo void
datos_estructurados = np.array(
    [(1, 'A', b'\x01\x02'), (2, 'B', b'\x03\x04')],
    dtype=[('id', 'i4'), ('nombre', 'U1'), ('datos_binarios', 'V2')]
)
print(datos_estructurados)
# Salida: [(1, 'A', b'\x01\x02') (2, 'B', b'\x03\x04')]
```
---
## Ejemplo de NumPy
```python
import numpy as np

# Ejemplo de varios tipos de datos en NumPy
int_np = np.array([1, 2, 3], dtype=np.int32)
uint_np = np.array([1, 2, 3], dtype=np.uint32)
float_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
complex_np = np.array([1+2j, 3+4j], dtype=np.complex64)
bool_np = np.array([True, False, True], dtype=bool)
string_np = np.array(["Hola", "Mundo"], dtype=np.string_)
object_np = np.array([1, "Hola", 3.14], dtype=object)


print(int_np, uint_np, float_np, complex_np, bool_np, string_np, , object_np, structured_np)
# Salida:
# [1 2 3] [1 2 3] [1. 2. 3.] [1.+2.j 3.+4.j] [ True False  True] [b'Hola' b'Mundo'] [1 'Hola' 3.14]
```
---
# Tipos de Datos en pandas

## Numéricos

### Enteros
- **`int8`**, **`int16`**, **`int32`**, **`int64`**: Enteros de 8, 16, 32 y 64 bits.

### Enteros sin signo
- **`uint8`**, **`uint16`**, **`uint32`**, **`uint64`**: Enteros sin signo de 8, 16, 32 y 64 bits.

### Flotantes
- **`float16`**, **`float32`**, **`float64`**: Números de punto flotante de 16, 32 y 64 bits.

---
## Ejemplo de Datos Numéricos
```python
import pandas as pd

# Ejemplo de Enteros
enteros_df = pd.DataFrame({'A': [1, 2, 3]}, dtype='int32')
print(enteros_df)
# Salida:
#    A
# 0  1
# 1  2
# 2  3

# Ejemplo de Enteros sin signo
enteros_sin_signo_df = pd.DataFrame({'A': [1, 2, 3]}, dtype='uint32')
print(enteros_sin_signo_df)
# Salida:
#    A
# 0  1
# 1  2
# 2  3

# Ejemplo de Flotantes
flotantes_df = pd.DataFrame({'A': [1.1, 2.2, 3.3]}, dtype='float64')
print(flotantes_df)
# Salida:
#      A
# 0  1.1
# 1  2.2
# 2  3.3
```
---
## Booleanos
- **`bool`**: Representa valores de `True` o `False`.

## Ejemplo de Booleanos
```python
import pandas as pd

booleanos_df = pd.DataFrame({'A': [True, False, True]}, dtype='bool')
print(booleanos_df)
# Salida:
#        A
# 0   True
# 1  False
# 2   True
```
---
## Cadenas de Caracteres
- **`object`**: Cadena que puede contener cualquier tipo de dato, pero generalmente se usa para cadenas de texto.
- **`string`**: Cadena específica para texto (disponible en versiones más recientes de pandas).
---
## Ejemplo de Cadenas de Caracteres
```python
import pandas as pd

# Ejemplo de Cadenas de Caracteres (object)
cadenas_df = pd.DataFrame({'A': ['Hola', 'Mundo']}, dtype='object')
print(cadenas_df)
# Salida:
#       A
# 0  Hola
# 1 Mundo

# Ejemplo de Cadenas de Texto (string)
cadenas_string_df = pd.DataFrame({'A': ['Hola', 'Mundo']}, dtype='string')
print(cadenas_string_df)
# Salida:
#       A
# 0  Hola
# 1 Mundo
```
---
## Fechas y Tiempos
- **`datetime64[ns]`**: Datos de fecha y hora con precisión de nanosegundos.

- **`datetime64[ns, tz]`**: Datos de fecha y hora con precisión de nanosegundos para zonas horarias.

- **`timedelta[ns]`**: Representa diferencias de tiempo con precisión de nanosegundos.

- **`period[freq]`**: Representa períodos, es decir, en intervalos de tiempo.
---
## Ejemplo de Fechas y Tiempos
```python
import pandas as pd

# Ejemplo de Fechas y Tiempos (datetime64[ns])
fechas_df = pd.DataFrame({'A': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01'])})
print(fechas_df)
# Salida:
#            A
# 0 2022-01-01
# 1 2022-02-01
# 2 2022-03-01

# Ejemplo de Fechas y Tiempos con Zona Horaria (datetime64[ns, tz])
fechas_tz_df = pd.DataFrame({'A': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']).tz_localize('UTC')})
print(fechas_tz_df)
# Salida:
#                          A
# 0 2022-01-01 00:00:00+00:00
# 1 2022-02-01 00:00:00+00:00
# 2 2022-03-01 00:00:00+00:00

# Ejemplo de Diferencias de Tiempo (timedelta[ns])
deltas_df = pd.DataFrame({'A': pd.to_timedelta(['1 days', '2 days', '3 days'])})
print(deltas_df)
# Salida:
#        A
# 0 1 days
# 1 2 days
# 2 3 days

# Ejemplo de Períodos (period[freq])
periodos_df = pd.DataFrame({'A': pd.period_range('2022-01', periods=3, freq='M')})
print(periodos_df)
# Salida:
#         A
# 0  2022-01
# 1  2022-02
# 2  2022-03
```
---
## Categóricos
- **`category`**: Representa datos categóricos que pueden ayudar a reducir el uso de memoria y mejorar la eficiencia.

## Ejemplo de Categórticos
```python
# Ejemplo de Datos Categóricos (category)
categoricos_df = pd.DataFrame({'A': pd.Categorical(['Tipo1', 'Tipo2', 'Tipo1'])})
print(categoricos_df)
# Salida:
#        A
# 0  Tipo1
# 1  Tipo2
# 2  Tipo1
```
---
## Intervalos
- **`Interval`**: Representa datos de intervalos.

## Ejemplo de Intervalos
```python
# Ejemplo de Intervalos (Interval)
intervalos_df = pd.DataFrame({'A': pd.interval_range(start=0, periods=3, freq=1)})
print(intervalos_df)
# Salida:
#             A
# 0  [0.0, 1.0)
# 1  [1.0, 2.0)
# 2  [2.0, 3.0)
```
---
## Nulos
- **`NA`**: Representa valores faltantes o nulos.

## Ejemplo de Nulos
```python
import pandas as pd
import numpy as np

# Ejemplo de Valores Nulos (NA)
nulos_df = pd.DataFrame({'A': [1, np.nan, 3]})
print(nulos_df)
# Salida:
#      A
# 0  1.0
# 1  NaN
# 2  3.0
```
---
# Conclusión
- **`Python`**: Tipos básicos como int, float, str.
- **`NumPy`**: Tipos numéricos optimizados para cálculos.
- **`Pandas`**: Tipos diseñados para manejar datos tabulares de manera eficiente.
---
# Fuentes

- **`Documentación de Python`** : https://docs.python.org/3/library/stdtypes.html
- **`Documentación de Numpy`** : https://numpy.org/doc/stable/user/basics.types.html
- **`Documentación de Pandas`**: https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#data-types
