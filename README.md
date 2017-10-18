# Instalación y uso de paquete powerflow

A continuación, se listan las instrucciones necesarias para la instalación y funcionamiento del paquete powerflow. 

## Creación de entorno virtual via virtualenv
1) Abrir un terminal

2) Instalar virtualenv via pip

a) Instalar pip (en caso de que no lo esté)

```sh
sudo easy_install pip
```

b) Instalar virtualenv

```sh
pip install virtualenv
```

3) Crear entorno virtual en directorio a elección*

```sh
virtualenv nombre_entorno_virtual
```

4) Para comenzar a utilizar el entorno virtual previamente creado, este debe ser activado:

```sh
source nombre_entorno_virtual/bin/activate
```

Al realizar este paso, aparecerá el nombre del entorno virtual en la parte izquierda del terminal, bajo el siguiente formato:

```sh
(nombre_entorno_virtual)nombre-equipo:nombre-proyecto nombre_usuario$
```

Lo que refleja que el entorno está activado. Así, todo paquete que sea instalado con pip estará en la carpeta creada para el entorno virtual.

5) En caso de que se quiera desactivar el entorno virtual, ejecutar el siguiente commando:

```sh
deactivate
```

*IMPORTANTE: Se recomienda no crear entorno virtual en el mismo directorio en que se ha clonado el repositorio. En caso de que el entorno virtual sea creado en dicho directorio, modificar archivo gitignore e incluirlo:

a) Abrir .gitignore

```sh
open -e .gitignore
```

b) Escribir en el archivo abierto la siguiente línea:

/nombre_entorno_virtual

c) Guardar cambios y cerrar archivo

## Creación de entorno virtual via virtualenvwrapper

De forma alternativa, es posible crear un entorno virtual por medio de virtualenvwrapper. Para ello, se requiere instalar virtualenv previamente. Luego:

1) En un terminal, instalar virtualenvwrapper

```sh
pip install virtualenvwrapper
```

2) Crear directorio que almacenará entornos virtuales creados

```sh
mkdir ~/.virtualenvs
```

3) Editar archivo .bash_profile y añadir las siguientes líneas:

```sh
export WORKON_HOME=~/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

4) Activar cambios realizados a archivo .bash_profile

```sh
source ~/.bash_profile
```

5) Crear entorno virtual, el cual utilizará python3 por defecto

```sh
mkvirtualenv --python=ruta_python3 nombre_entorno_virtual
```

Por ejemplo:

```sh
mkvirtualenv --python=/usr/local/bin/python3 nombre_entorno_virtual
```

Por defecto, el entorno virtual quedará inmediatamente activado posterior a su creación.

6) Para activar entorno virtual previamente creado, se debe ejecutar el siguiente comando:

```sh
workon nombre_entorno_virtual
```

7) Para desactivar entorno virtual, ejecutar el siguiente comando:

```sh
deactivate
```

## Instalación de paquete powerflow

1) En un terminal, situarse dentro de la carpeta principal del repositorio

2) Instalar requerimientos:

```sh
pip install -r requirements.txt
```

3) Ejecutar el siguiente comando, el cual instalará el paquete (notar el punto a la derecha de la opción -e):

```sh
pip install -e .
```

## Uso de paquete powerflow

1) Para la ejecución de una determinada funcionalidad del paquete, se debe escribir en el terminal:

```sh
powerflow <command> <param_1 param_2 ... param_n> <--arg_1 --arg_2 ... --arg_n>
```

Donde `command` es el comando que se desea ejecutar, `param_i` es el i-ésimo parámetro recibido por command y `arg_i` es el i-ésimo argumento opcional asociado a command.

Por ejemplo, la siguiente línea:

```sh
powerflow ybc ruta_bus_data ruta_branch_data nombre --force=True
```

Permite la ejecución del comando `ybc`, el que recibe a `ruta_bus_data`, `ruta_branch_data` y `nombre` como parámetros y a `force` como argumento opcional.

2) Para obtener información acerca de los comandos que pueden ser utilizados, escribir en un terminal:

```sh
powerflow --help
```

3) Para obtener información acerca del funcionamiento de un comando en específico, escribir en un terminal:

```sh
powerflow <command> --help
```

## Testing de funcionalidades

1) Para la ejecución de pruebas sobre el paquete powerflow, se encuentran disponibles scripts de test en el directorio `tests` del repositorio. Para la ejecución de un test determinado, situarse dentro de dicha carpeta y ejecutar la siguiente línea en el terminal:

```sh
pytest
```

2) Notar que en el script `tests/test_loadflow.py` se comprobaron sólo las tensiones (`bus_solution`) y la convergencia del algoritmo. En consecuencia, queda pendiente testear los flujos del sistema asociados a líneas, transformadores y equipos shunt (`line_flows`, `transformer_flows` y `shunt_flows`, respectivamente).

3) Archivos de prueba `bus_data.csv` y `branch_data.csv` se encuentran en el directorio `tests/PowerFlow/FastYBus`, que son los mismos archivos utilizados para ejecutar testear el algoritmo. Estos archivos se encuentran clasificados segun los casos IEEE de 14, 30, 57, 118 y 300 barras.