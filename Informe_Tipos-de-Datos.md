# Informe sobre Tipos de Datos en Python, NumPy y Pandas

## Introducción

En el ámbito de la programación y el análisis de datos, comprender los tipos de datos es fundamental para el manejo eficiente de la información. Este informe explora los tipos de datos en Python, así como en las bibliotecas NumPy y Pandas, destacando sus características, ejemplos y limitaciones.

## Tipos de Datos en Python

### 1. Tipos Numéricos

#### Enteros (`int`)
- **Descripción**: Representan números enteros, tanto positivos como negativos, sin decimales.
- **Ejemplo**:
    ```python
  edad = 25
  print(edad)  # Salida: 25
  ```

- **Limitaciones**: En Python, los enteros son de precisión arbitraria, lo que significa que pueden crecer según la memoria disponible. Sin embargo, en otros lenguajes, los enteros pueden estar limitados por el tamaño de su tipo (por ejemplo, int32 o int64).

#### Flotantes (`float`)
- **Descripción**: Representan números de punto flotante, que pueden tener una parte decimal.
- **Ejemplo**:
    ```python
    altura = 1.75
    print(altura) # Salida: 1.75
    ```

- **Limitaciones**: Los flotantes tienen precisión limitada y pueden introducir errores de redondeo en cálculos. Esto es especialmente relevante en operaciones matemáticas complejas.

#### Complejos (`complex`)
- **Descripción**: Representan números complejos, que tienen una parte real y una parte imaginaria.
- **Ejemplo**:
    ```python
    numero_complejo = 2 + 3j
    print(numero_complejo) # Salida: (2+3j)
    ```

- **Limitaciones**: Aunque Python maneja números complejos, no todos los lenguajes de programación lo hacen, lo que puede limitar la portabilidad del código.

### 2. Secuencias

#### Cadenas de Texto (`str`)
- **Descripción**: Secuencias de caracteres que representan texto.
- **Ejemplo**:
    ```python
    nombre = "Ana"
    print(nombre)  # Salida: Ana
    ```
#### Listas (`list`)
- **Descripción**: Colecciones ordenadas y mutables que pueden contener elementos de diferentes tipos.
- **Ejemplo**:
    ```python
    numeros = [1, 2, 3, 4, 5]
    print(numeros)  # Salida: [1, 2, 3, 4, 5]
    ```

- **Limitaciones**: Las listas pueden consumir más memoria que otros tipos de datos debido a su flexibilidad. Además, su rendimiento puede verse afectado en operaciones de búsqueda y ordenación.

#### Tuplas (`tuple`)
- **Descripción**: Colecciones ordenadas e inmutables, similares a las listas.
- **Ejemplo**:
    ```python
    coordenadas = (10, 20)
    print(coordenadas)  # Salida: (10, 20)
    ```

**Limitaciones**: Al ser inmutables, no se pueden modificar después de su creación, lo que puede ser una desventaja si se requiere flexibilidad.

### 3. Conjuntos

#### Conjuntos (`set`)
- **Descripción**: Colecciones no ordenadas de elementos únicos.
- **Ejemplo**:
    ```python
    frutas = {"manzana", "banana", "naranja"}
    print(frutas) # Salida: {'banana', 'naranja', 'manzana'}
    ```

- **Limitaciones**: No mantienen el orden de los elementos y no permiten duplicados. Esto puede ser problemático si se necesita conservar el orden de inserción.

#### Conjuntos Inmutables (`frozenset`)
- **Descripción**: Versiones inmutables de los conjuntos.
- **Ejemplo**:
    ```python
    frutas_inmutables = frozenset(["manzana", "banana", "naranja"])
    print(frutas_inmutables) # Salida: frozenset({'banana', 'naranja', 'manzana'})
    ```

- **Limitaciones**: Al ser inmutables, no se pueden modificar después de su creación, lo que limita su uso en ciertas situaciones.

### 4. Diccionarios (`dict`)
- **Descripción**: Colecciones de pares clave-valor que permiten el acceso rápido a los valores a través de sus claves.
- **Ejemplo**:
    ```python
    persona = {"nombre": "Ana", "edad": 25}
    print(persona) # Salida: {'nombre': 'Ana', 'edad': 25}
    ```

- **Limitaciones**: Los diccionarios pueden consumir más memoria que las listas y su rendimiento puede verse afectado en operaciones de búsqueda si las claves no son únicas.

### 5. Booleanos (`bool`)
- **Descripción**: Representan valores de verdad, `True` o `False`.
- **Ejemplo**:
    ```python
    es_mayor_de_edad = True
    print(es_mayor_de_edad) # Salida: True
    ```

- **Limitaciones**: Los booleanos son simples y no tienen limitaciones significativas, pero su uso puede ser confuso en expresiones complejas.

### 6. NoneType
- **Descripción**: Representa la ausencia de valor o un valor nulo.
- **Ejemplo**:
    ```python
    valor_nulo = None
    print(valor_nulo) # Salida: None
    ```

- **Limitaciones**: Puede ser confuso en el manejo de errores y en la verificación de valores, ya que puede ser confundido con otros tipos de datos.

## Tipos de Datos en NumPy

NumPy es una biblioteca fundamental para la computación científica en Python, que proporciona tipos de datos optimizados para cálculos numéricos.

### 1. Tipos Numéricos

#### Enteros
- **Ejemplos**: `np.int8`, `np.int16`, `np.int32`, `np.int64`.
- **Limitaciones**: Cada tipo tiene un rango específico. Por ejemplo, `np.int8` solo puede almacenar valores entre -128 y 127,
`np.int16` solo puede almacenar valores entre -32,768 a 32,767,
`np.int32` solo puede almacenar valores entre -2,147,483,648 a 2,147,483,647,
`np.int32` solo puede almacenar valores entre -2,147,483,648 a 2,147,483,647,
`np.int64` solo puede almacenar valores entre -9,223,372,036,854,775,808 a 9,223,372,036,854,775,807.

#### Flotantes
- **Ejemplos**: `np.float16`, `np.float32`, `np.float64`.
- **Limitaciones**: Al igual que en Python, los flotantes pueden sufrir problemas de precisión y redondeo, por ejemplo, la precisión de `np.float16`su precisión es de alrededor de 3 a 4 dígitos decimales, `np.float16`su precisión es de alrededor de 7 dígitos decimales, `np.float16` su precisión es de alrededor de de 15 a 16 dígitos decimales.
#### Complejos
- **Ejemplos**: `np.complex64`, `np.complex128`.
- **Limitaciones**: El uso de números complejos puede ser limitado en algunos contextos, especialmente en bibliotecas que no los soportan.

### 2. Booleanos
- **Descripción**: Representan valores de `True` o `False`.
- **Ejemplo**:
    ```python
    import numpy as np
    booleanos = np.array([True, False, True], dtype=bool)
    print(booleanos) # Salida: [ True False True]
    ```

- **Limitaciones**: Similar a los booleanos en Python, su uso puede ser confuso en expresiones complejas.

### 3. Cadenas de Caracteres
- **Descripción**: `string_` para cadenas de longitud fija.
- **Ejemplo**:
    ```python
    import numpy as np
    cadenas = np.array(["Hola", "Mundo"], dtype=np.string_)
    print(cadenas) # Salida: [b'Hola' b'Mundo']
    ```

- **Limitaciones**: Las cadenas de longitud fija pueden ser menos flexibles que las cadenas de Python.

### 4. Objetos
- **Descripción**: `object_` puede contener cualquier objeto de Python.
- **Ejemplo**:
    ```python
    import numpy as np
    objetos = np.array([1, "Hola", 3.14], dtype=object)
    print(objetos)  # Salida: [1 'Hola' 3.14]
    ```

- **Limitaciones**: El uso de `object_` puede llevar a un rendimiento más lento y un mayor consumo de memoria.

### 5. Datos Estructurados
- **Descripción**: `void` permite crear arreglos con tipos de datos personalizados.
- **Ejemplo**:
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

- **Limitaciones**: La complejidad de los datos estructurados puede dificultar su manipulación y análisis.

## Tipos de Datos en Pandas

Pandas es una biblioteca que proporciona estructuras de datos y herramientas de análisis de datos.

### 1. Tipos Numéricos

#### Enteros
- **Ejemplos**: `int8`, `int16`, `int32`, `int64`.
- **Limitaciones**: Al igual que en NumPy, cada tipo tiene un rango específico.

#### Flotantes
- **Ejemplos**: `float16`, `float32`, `float64`.
- **Limitaciones**: Pueden sufrir problemas de precisión y redondeo.

### Booleanos
- **Descripción**: Representan valores de `True` o `False`.
- **Ejemplo**:
    ```python
    import pandas as pd
    booleanos_df = pd.DataFrame({'A': [True, False, True]}, dtype='bool')
    print(booleanos_df)
    ```

- **Limitaciones**: Similar a los booleanos en Python y NumPy.

### 3. Cadenas de Caracteres
- **Descripción**: `object` para cadenas de texto y `string` para cadenas específicas.
- **Ejemplo**:
    ```python
    import pandas as pd
    cadenas_df = pd.DataFrame({'A': ['Hola', 'Mundo']}, dtype='object')
    print(cadenas_df)
    ```
text
- **Limitaciones**: El tipo `object` puede ser menos eficiente en términos de memoria.

### 4. Fechas y Tiempos
- **Descripción**: `datetime64[ns]` para datos de fecha y hora.
- **Ejemplo**:
    ```python
    import pandas as pd
    fechas_df = pd.DataFrame({'A': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01'])})
    print(fechas_df)
    ```

- **Limitaciones**: La manipulación de fechas y horas puede ser compleja y propensa a errores.

### 5. Categóricos
- **Descripción**: `category` para datos categóricos que ayudan a reducir el uso de memoria.
- **Ejemplo**:
    ```python
    import pandas as pd
    categoricos_df = pd.DataFrame({'A': pd.Categorical(['Tipo1', 'Tipo2', 'Tipo1'])})
    print(categoricos_df)
    ```

- **Limitaciones**: La conversión de datos a categóricos puede ser costosa en términos de tiempo y puede no ser adecuada para todos los conjuntos de datos.

### 6. Intervalos
- **Descripción**: `Interval` para representar datos de intervalos.
- **Ejemplo**:
    ```python
    import pandas as pd
    intervalos_df = pd.DataFrame({'A': pd.interval_range(start=0, periods=3, freq=1)})
    print(intervalos_df)
    ```

- **Limitaciones**: La manipulación de intervalos puede ser menos intuitiva y puede requerir un manejo especial en análisis.

### 7. Nulos
- **Descripción**: `NA` para representar valores faltantes o nulos.
- **Ejemplo**:
    ```python
    import pandas as pd
    import numpy as np
    nulos_df = pd.DataFrame({'A': [1, np.nan, 3]})
    print(nulos_df)
    ```

- **Limitaciones**: El manejo de valores nulos puede complicar el análisis y requerir limpieza de datos.

## Conclusión

La comprensión de los tipos de datos en Python, NumPy y Pandas es esencial para el desarrollo de aplicaciones eficientes y el análisis de datos. Cada tipo de dato tiene sus propias características y limitaciones, lo que influye en su uso en diferentes contextos. Al elegir un tipo de dato, es importante considerar tanto las necesidades de la aplicación como las limitaciones inherentes a cada tipo.

# Fuentes

- **`Documentación de Python`** : https://docs.python.org/3/library/stdtypes.html
- **`Documentación de Numpy`** : https://numpy.org/doc/stable/user/basics.types.html
- **`Documentación de Pandas`**: https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#data-types