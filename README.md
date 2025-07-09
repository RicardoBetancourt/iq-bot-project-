### 1. Crear un Repositorio en GitHub

1. **Inicia sesión en GitHub**: Ve a [GitHub](https://github.com) y accede a tu cuenta.

2. **Crear un nuevo repositorio**:
   - Haz clic en el botón "+" en la esquina superior derecha y selecciona "New repository".
   - Asigna un nombre a tu repositorio (por ejemplo, `iq-trading-bot`).
   - Opcionalmente, agrega una descripción.
   - Selecciona si deseas que el repositorio sea público o privado.
   - Puedes inicializar el repositorio con un archivo README si lo deseas (esto es opcional).
   - Haz clic en "Create repository".

### 2. Subir Archivos a tu Repositorio

#### Opción 1: Usar la Interfaz Web de GitHub

1. **Accede a tu nuevo repositorio**.
2. Haz clic en el botón "Add file" y selecciona "Upload files".
3. Arrastra y suelta los archivos `iq.py` y `prueba.py` en el área de carga o haz clic en "choose your files" para seleccionarlos desde tu computadora.
4. Una vez que los archivos estén listos para ser subidos, desplázate hacia abajo y agrega un mensaje de confirmación (commit message).
5. Haz clic en "Commit changes".

#### Opción 2: Usar Git en la Línea de Comandos

Si prefieres usar la línea de comandos, sigue estos pasos:

1. **Instala Git**: Si no tienes Git instalado, descárgalo e instálalo desde [git-scm.com](https://git-scm.com/).

2. **Clona el repositorio**:
   Abre una terminal y ejecuta el siguiente comando (reemplaza `USERNAME` con tu nombre de usuario de GitHub y `REPO_NAME` con el nombre de tu repositorio):
   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   ```

3. **Navega al directorio del repositorio**:
   ```bash
   cd REPO_NAME
   ```

4. **Copia tus archivos al directorio del repositorio**:
   Asegúrate de que `iq.py` y `prueba.py` estén en el directorio del repositorio.

5. **Agrega los archivos al repositorio**:
   ```bash
   git add iq.py prueba.py
   ```

6. **Confirma los cambios**:
   ```bash
   git commit -m "Agregar iq.py y prueba.py"
   ```

7. **Sube los cambios a GitHub**:
   ```bash
   git push origin main
   ```

   Nota: Si tu rama principal se llama `master`, usa `master` en lugar de `main`.

### 3. Verifica que los Archivos se Subieron Correctamente

1. Regresa a tu repositorio en GitHub y verifica que los archivos `iq.py` y `prueba.py` estén listados.

¡Y eso es todo! Ahora tienes un repositorio en GitHub con tus archivos `iq.py` y `prueba.py`.