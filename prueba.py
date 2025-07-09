from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from pathlib import Path
import random
import os
import matplotlib
matplotlib.use('Agg')  # Usa el backend que no requiere interfaz gráfica
import matplotlib.pyplot as plt

# ----------------------------
# Configuración inicial
# ----------------------------

def configurar_logging():
    """Configura el sistema de logging"""
    try:
        directorio_log = Path.home() / "IQOption_Logs"
        directorio_log.mkdir(exist_ok=True)
        
        archivo_log = directorio_log / f"bot_trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(archivo_log, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Logging configurado correctamente")
        return True
        
    except Exception as e:
        print(f"Error al configurar logging: {str(e)}")
        return False

if not configurar_logging():
    exit("Error en configuración inicial")

# ----------------------------
# Clase Backtester
# ----------------------------

class Backtester:
    def __init__(self, datos_historicos, activo):
        self.datos = datos_historicos
        self.activo = activo
        self.resultados = []
        self.estadisticas = {}
        
    def ejecutar_backtest(self, porcentaje_entrenamiento=0.7):
        """Ejecuta el backtesting completo"""
        try:
            logging.info(f"Iniciando backtest para {self.activo}")
            
            # Dividir datos en entrenamiento y prueba
            split_index = int(len(self.datos) * porcentaje_entrenamiento)
            train_data = self.datos.iloc[:split_index]
            test_data = self.datos.iloc[split_index:]
            
            # Entrenar modelo
            modelo = self.entrenar_modelo(train_data)
            
            if modelo is None:
                return False
                
            # Probar modelo
            self.probar_modelo(modelo, test_data)
            
            # Calcular estadísticas
            self.calcular_estadisticas()
            
            # Generar reporte
            self.generar_reporte()
            
            return True
            
        except Exception as e:
            logging.error(f"Error en backtesting: {str(e)}")
            return False
    
    def entrenar_modelo(self, datos_entrenamiento):
        """Entrena el modelo predictivo con SMOTE"""
        try:
            features = ['close', 'SMA_10', 'EMA_12', 'RSI', 'MACD', 'Signal',
                      'SMA_20', 'BB_upper', 'BB_lower', 'Momentum_5', 'Volatilidad']
            X = datos_entrenamiento[features]
            y = datos_entrenamiento['Objetivo']
            
            # Balancear clases con SMOTE
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
            
            modelo = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            modelo.fit(X_res, y_res)
            
            train_acc = accuracy_score(y_res, modelo.predict(X_res))
            logging.info(f"Modelo entrenado - Precisión: {train_acc:.2%}")
            
            return modelo
            
        except Exception as e:
            logging.error(f"Error entrenando modelo: {str(e)}")
            return None
    
    def probar_modelo(self, modelo, datos_prueba):
        """Prueba el modelo con datos no vistos"""
        try:
            features = ['close', 'SMA_10', 'EMA_12', 'RSI', 'MACD', 'Signal',
                      'SMA_20', 'BB_upper', 'BB_lower', 'Momentum_5', 'Volatilidad']
            X_test = datos_prueba[features]
            y_test = datos_prueba['Objetivo']
            
            # Predecir y evaluar
            predicciones = modelo.predict(X_test)
            probas = modelo.predict_proba(X_test)
            
            # Simular operaciones con gestión de riesgo mejorada
            balance = 10000  # Balance inicial simulado
            monto_base = 100  # Monto base por operación
            monto_actual = monto_base
            perdidas_consecutivas = 0
            max_perdidas_consecutivas = 30
            multiplicador = 3.0
            
            resultados = []
            
            for i in range(len(predicciones)):
                # Gestión de capital tipo Martingala modificada
                if perdidas_consecutivas >= max_perdidas_consecutivas:
                    monto_actual = monto_base
                    perdidas_consecutivas = 0
                
                direccion = 'call' if predicciones[i] == 1 else 'put'
                probabilidad = probas[i][1] if direccion == 'call' else probas[i][0]
                
                # Simular resultado
                resultado_real = y_test.iloc[i]
                
                # Calcular ganancia (80% para win, 100% loss)
                if (direccion == 'call' and resultado_real == 1) or (direccion == 'put' and resultado_real == 0):
                    ganancia = monto_actual * 0.8
                    balance += ganancia
                    resultado = 'ganada'
                    monto_actual = monto_base  # Resetear después de ganar
                    perdidas_consecutivas = 0
                else:
                    ganancia = -monto_actual
                    balance += ganancia
                    resultado = 'perdida'
                    perdidas_consecutivas += 1
                    monto_actual *= multiplicador
                
                resultados.append({
                    'fecha': datos_prueba.index[i],
                    'direccion': direccion,
                    'probabilidad': probabilidad,
                    'resultado': resultado,
                    'ganancia': ganancia,
                    'balance': balance,
                    'monto_operacion': monto_actual
                })
            
            self.resultados = resultados
            return True
            
        except Exception as e:
            logging.error(f"Error probando modelo: {str(e)}")
            return False
    
    def calcular_estadisticas(self):
        """Calcula estadísticas de rendimiento mejoradas"""
        if not self.resultados:
            return
            
        df = pd.DataFrame(self.resultados)
        
        # Operaciones ganadas vs perdidas
        ganadas = df[df['resultado'] == 'ganada']
        perdidas = df[df['resultado'] == 'perdida']
        
        # Ratio de ganancias
        win_rate = len(ganadas) / len(df)
        
        # Profit Factor
        ganancia_total = ganadas['ganancia'].sum()
        perdida_total = abs(perdidas['ganancia'].sum())
        profit_factor = ganancia_total / perdida_total if perdida_total > 0 else float('inf')
        
        # Drawdown
        df['balance_acum'] = df['ganancia'].cumsum() + 10000
        max_balance = df['balance_acum'].cummax()
        df['drawdown'] = (max_balance - df['balance_acum']) / max_balance
        max_drawdown = df['drawdown'].max()
        
        # Sharpe Ratio (simplificado)
        retorno_medio = df['ganancia'].mean()
        volatilidad = df['ganancia'].std()
        sharpe_ratio = retorno_medio / volatilidad if volatilidad != 0 else 0
        
        # Expectativa matemática
        expectativa = (win_rate * ganadas['ganancia'].mean()) + ((1-win_rate) * perdidas['ganancia'].mean())
        
        self.estadisticas = {
            'total_operaciones': len(df),
            'operaciones_ganadas': len(ganadas),
            'operaciones_perdidas': len(perdidas),
            'win_rate': win_rate,
            'ganancia_total': ganancia_total,
            'perdida_total': perdida_total,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'balance_final': df['balance'].iloc[-1],
            'sharpe_ratio': sharpe_ratio,
            'expectativa': expectativa,
            'retorno_porcentual': (df['balance'].iloc[-1] - 10000) / 10000
        }
    
    def generar_reporte(self):
        """Genera un reporte visual y textual del backtesting"""
        try:
            if not self.resultados:
                return
                
            df = pd.DataFrame(self.resultados)
            
            # Crear directorio para reportes
            directorio_reportes = Path.home() / "IQOption_Reportes"
            directorio_reportes.mkdir(exist_ok=True)
            
            # Gráfico de balance
            plt.figure(figsize=(14, 7))
            plt.plot(df['fecha'], df['balance'], label='Balance', color='blue')
            
            # Marcar operaciones ganadas y perdidas
            ganadas = df[df['resultado'] == 'ganada']
            perdidas = df[df['resultado'] == 'perdida']
            plt.scatter(ganadas['fecha'], ganadas['balance'], color='green', label='Ganadas', alpha=0.5)
            plt.scatter(perdidas['fecha'], perdidas['balance'], color='red', label='Perdidas', alpha=0.5)
            
            plt.title(f'Evolución del Balance - {self.activo}')
            plt.xlabel('Fecha')
            plt.ylabel('Balance ($)')
            plt.legend()
            plt.grid(True)
            
            archivo_reporte = directorio_reportes / f"backtest_{self.activo}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(archivo_reporte)
            plt.close()
            
            # Reporte textual mejorado
            reporte = f"""
            ============ REPORTE BACKTESTING ============
            Activo: {self.activo}
            Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Período: {df['fecha'].iloc[0]} a {df['fecha'].iloc[-1]}
            ---------------------------------------------
            ESTADÍSTICAS DE OPERACIONES:
            Total operaciones: {self.estadisticas['total_operaciones']}
            Operaciones ganadas: {self.estadisticas['operaciones_ganadas']} ({self.estadisticas['win_rate']:.2%})
            Operaciones perdidas: {self.estadisticas['operaciones_perdidas']}
            ---------------------------------------------
            RENDIMIENTO:
            Ganancia total: ${self.estadisticas['ganancia_total']:.2f}
            Pérdida total: ${self.estadisticas['perdida_total']:.2f}
            Profit Factor: {self.estadisticas['profit_factor']:.2f}
            Retorno total: {self.estadisticas['retorno_porcentual']:.2%}
            Expectativa por operación: ${self.estadisticas['expectativa']:.2f}
            ---------------------------------------------
            RIESGO:
            Máximo Drawdown: {self.estadisticas['max_drawdown']:.2%}
            Sharpe Ratio: {self.estadisticas['sharpe_ratio']:.2f}
            ---------------------------------------------
            BALANCE FINAL: ${self.estadisticas['balance_final']:.2f}
            =============================================
            """
            
            logging.info(reporte)
            
            # Guardar reporte en archivo
            archivo_texto = directorio_reportes / f"reporte_{self.activo}_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(archivo_texto, 'w') as f:
                f.write(reporte)
                
            return True
            
        except Exception as e:
            logging.error(f"Error generando reporte: {str(e)}")
            return False

# ----------------------------
# Clase principal del Bot
# ----------------------------

class BotIQTrading:
    def __init__(self, email, password, account_type="PRACTICE", **kwargs):
        try:
            # Configuración básica
            self.email = email
            self.password = password
            self.account_type = account_type
            self.activos = kwargs.get('activos', ["EURUSD-OTC", "GBPUSD-OTC", "EURGBP-OTC"])
            self.activo_actual = None
            self.tipo_cuenta = kwargs.get('tipo_cuenta', "PRACTICE")
            
            # Estrategia
            self.monto_inicial = 1.0
            self.monto_actual = self.monto_inicial
            self.perdidas_consecutivas = 0
            self.max_perdidas_consecutivas = 30
            self.multiplicador = 3.0
            
            # Directorios
            self.directorio_base = Path.home() / "IQOption_Trading"
            self.directorio_historial = self.directorio_base / "historial_operaciones"
            self.directorio_reportes = self.directorio_base / "reportes_backtesting"
            
            # Crear directorios necesarios
            self.crear_directorios()
            
            # Estado
            self.api = None
            self.conectado = False
            self.modelos = {}
            self.datos_historicos = {}
            self.operacion_activa = False
            self.id_operacion_actual = None
            self.historial_operaciones = []
            self.balance = 0
            self.ejecutando = True
            self.ultima_prediccion = None
            
            # Tiempos
            self.ultimo_tiempo_operacion = 0
            self.tiempo_espera = 30
            
            # Riesgo
            self.max_operaciones_diarias = 10000
            self.stop_loss_diario = 1.0
            self.take_profit_diario = 1.0
            self.operaciones_hoy = 0
            self.balance_inicial_dia = 0
            
            logging.info("Bot inicializado correctamente")
            
        except Exception as e:
            logging.error(f"Error al inicializar: {str(e)}")
            raise

    def crear_directorios(self):
        """Crea los directorios necesarios para el funcionamiento del bot"""
        try:
            self.directorio_base.mkdir(exist_ok=True)
            self.directorio_historial.mkdir(exist_ok=True)
            self.directorio_reportes.mkdir(exist_ok=True)
            logging.info("Directorios creados correctamente")
        except Exception as e:
            logging.error(f"Error creando directorios: {str(e)}")
            raise

    def verificar_activos_disponibles(self):
        """Verifica qué activos están realmente disponibles para operar"""
        disponibles = []
        try:
            logging.info("Verificando activos disponibles...")
            temp_api = IQ_Option(self.email, self.password)
            if temp_api.connect():
                for activo in self.activos:
                    try:
                        velas = temp_api.get_candles(activo, 60, 10, time.time())
                        if velas:
                            disponibles.append(activo)
                            logging.info(f"Activo {activo} disponible")
                        else:
                            logging.warning(f"Activo {activo} no disponible")
                    except Exception as e:
                        logging.warning(f"Error verificando activo {activo}: {str(e)}")
                temp_api.disconnect()
            else:
                logging.warning("No se pudo conectar para verificar activos")
                return self.activos  # Si no podemos verificar, usamos todos
            
            return disponibles if disponibles else self.activos
        except Exception as e:
            logging.error(f"Error verificando activos: {str(e)}")
            return self.activos

    def conectar_api(self):
        """Conecta con la API de IQ Option"""
        try:
            logging.info("Conectando a IQ Option...")
            self.api = IQ_Option(self.email, self.password)
            
            for intento in range(3):
                try:
                    exito, razon = self.api.connect()
                    if exito:
                        # Cambia el tipo de cuenta según el valor recibido
                        self.api.change_balance(self.account_type)
                        self.conectado = True
                        self.balance = self.api.get_balance()
                        self.balance_inicial_dia = self.balance
                        logging.info(f"Conectado. Balance: ${self.balance:.2f}")
                        return True
                    
                    logging.warning(f"Intento {intento+1} fallido: {razon}")
                    time.sleep(5)
                
                except Exception as e:
                    logging.warning(f"Error en conexión: {str(e)}")
                    time.sleep(10)
            
            return False
            
        except Exception as e:
            logging.error(f"Error crítico: {str(e)}")
            return False

    def configurar_tipo_cuenta(self):
        """Configura el tipo de cuenta (PRACTICE o REAL)"""
        try:
            if self.api.get_balance_mode() == self.tipo_cuenta:
                return True
                
            if self.api.change_balance(self.tipo_cuenta):
                return self.api.get_balance_mode() == self.tipo_cuenta
            return False
        except Exception as e:
            logging.error(f"Error configurando cuenta: {str(e)}")
            return False

    def obtener_datos_historicos(self, activo, cantidad_velas=1000, return_df=False):
        """Obtiene datos históricos para un activo específico. Si return_df=True, retorna el DataFrame procesado o None."""
        try:
            logging.info(f"Obteniendo datos para {activo} (30s)")
            # Cambiado a timeframe de 30 segundos
            velas = self.api.get_candles(activo, 30, cantidad_velas, time.time())
            if velas and len(velas) > 100:
                df = self.procesar_datos_mercado(velas, activo)
                self.datos_historicos[activo] = df
                if return_df:
                    return df
                return True
            logging.warning(f"Datos insuficientes para {activo}")
            if return_df:
                return None
            return False
        except Exception as e:
            logging.error(f"Error obteniendo datos para {activo}: {str(e)}")
            if return_df:
                return None
            return False

    def procesar_datos_mercado(self, velas, activo):
        """Procesa los datos del mercado y calcula indicadores técnicos, incluyendo ADX"""
        try:
            df = pd.DataFrame(velas)
            df['fecha'] = pd.to_datetime(df['from'], unit='s')
            df.set_index('fecha', inplace=True)
            df = df[['open', 'max', 'min', 'close', 'volume']]
            # Mantener nombres en inglés para compatibilidad con iq.py
            df.columns = ['open', 'high', 'low', 'close', 'volume']


            # Indicadores técnicos básicos (sin ADX)
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain/loss)))

            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['STD_20'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['SMA_20'] + (df['STD_20'] * 2)
            df['BB_lower'] = df['SMA_20'] - (df['STD_20'] * 2)

            # Momentum
            df['Momentum_5'] = df['close'].pct_change(5)

            # Volatilidad
            df['Volatilidad'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()

            # Objetivo (1 si el próximo close es mayor, 0 si es menor)
            df['Objetivo'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            df.dropna(inplace=True)

            logging.info(f"Datos procesados para {activo}. Total registros: {len(df)}")
            return df

        except Exception as e:
            logging.error(f"Error procesando datos para {activo}: {str(e)}")
            raise

    def ejecutar_backtesting_activos(self):
        """Ejecuta backtesting para todos los activos configurados"""
        try:
            logging.info("Iniciando backtesting para todos los activos")
            
            # Verificar activos disponibles primero
            self.activos = self.verificar_activos_disponibles()
            
            if not self.activos:
                logging.error("Ningún activo disponible para operar")
                return False
                
            for activo in self.activos:
                if not self.obtener_datos_historicos(activo, 1000):
                    continue
                    
                backtester = Backtester(self.datos_historicos[activo], activo)
                if backtester.ejecutar_backtest():
                    stats = backtester.estadisticas
                    
                    # Criterios flexibles para aceptar activo
                    if stats.get('win_rate', 0) > 0.51 or stats.get('profit_factor', 0) > 0.85:
                        self.entrenar_modelo_activo(activo)
                        logging.info(f"Activo {activo} aceptado para trading")
                    else:
                        logging.warning(f"Activo {activo} no cumple criterios mínimos")
            
            return True
            
        except Exception as e:
            logging.error(f"Error en backtesting general: {str(e)}")
            return False

    def entrenar_modelo_activo(self, activo):
        """Entrena modelo para un activo específico"""
        try:
            if activo not in self.datos_historicos:
                logging.warning(f"No hay datos para entrenar modelo de {activo}")
                return False
                
            df = self.datos_historicos[activo]
            features = ['close', 'SMA_10', 'EMA_12', 'RSI', 'MACD', 'Signal',
                      'SMA_20', 'BB_upper', 'BB_lower', 'Momentum_5', 'Volatilidad']
            X = df[features]
            y = df['Objetivo']
            
            # Balancear clases con SMOTE
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
            
            modelo = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            modelo.fit(X_res, y_res)
            
            self.modelos[activo] = modelo
            logging.info(f"Modelo entrenado para {activo}")
            return True
            
        except Exception as e:
            logging.error(f"Error entrenando modelo para {activo}: {str(e)}")
            return False

    def seleccionar_condiciones_por_activo(self, activo):
        """Devuelve las condiciones de trading para un activo específico, o None si no se puede calcular."""
        try:
            if not self.modelos or activo not in self.modelos or activo not in self.datos_historicos:
                return None
            modelo = self.modelos[activo]
            ultima_vela = self.datos_historicos[activo].iloc[-1]
            features = ['close', 'SMA_10', 'EMA_12', 'RSI', 'MACD', 'Signal',
                        'SMA_20', 'BB_upper', 'BB_lower', 'Momentum_5', 'Volatilidad']
            X = pd.DataFrame([ultima_vela[features]])
            pred = modelo.predict(X)[0]
            proba = modelo.predict_proba(X)[0]
            probabilidad = proba[1] if pred == 1 else proba[0]

            # --- EMA Trend Logic ---
            # Calculate EMA(3) and EMA(20) for the last 5 candles for trend confirmation
            df = self.datos_historicos[activo]
            if len(df) < 25:
                return None
            df['EMA_3'] = df['close'].ewm(span=3, adjust=False).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            last5 = df.iloc[-5:]
            ema3_above_ema20 = (last5['EMA_3'] > last5['EMA_20']).all()
            ema3_below_ema20 = (last5['EMA_3'] < last5['EMA_20']).all()

            if ema3_above_ema20:
                tendencia = 'bullish'
            elif ema3_below_ema20:
                tendencia = 'bearish'
            else:
                tendencia = 'lateral'

            # Solo operar en dirección de la tendencia, nunca en lateral
            if tendencia == 'lateral':
                return None
            direccion = 'call' if tendencia == 'bullish' else 'put'
            if (direccion == 'call' and pred != 1) or (direccion == 'put' and pred != 0):
                # Solo operar si la predicción del modelo coincide con la tendencia EMA
                return None

            condiciones = {
                'activo': activo,
                'direccion': direccion,
                'probabilidad': probabilidad,
                'rsi': ultima_vela['RSI'],
                'precio': ultima_vela['close'],
                'bb_upper': ultima_vela['BB_upper'],
                'bb_lower': ultima_vela['BB_lower'],
                'ema3': ultima_vela.get('EMA_3', np.nan),
                'ema20': ultima_vela.get('EMA_20', np.nan),
                'tendencia_ema': tendencia
            }
            # Filtro adicional: precio cerca de una banda de Bollinger
            precio = condiciones['precio']
            bb_upper = condiciones['bb_upper']
            bb_lower = condiciones['bb_lower']
            rango_bb = bb_upper - bb_lower
            if (precio > bb_upper - 0.25*rango_bb) or (precio < bb_lower + 0.25*rango_bb):
                condiciones['probabilidad'] *= 1.1  # Aumentar probabilidad en zonas clave
            return condiciones
        except Exception as e:
            logging.warning(f"Error evaluando {activo} (EMA trend): {str(e)}")
            return None

    def ejecutar_operacion(self, activo, direccion):
        """Ejecuta una operación en el activo especificado"""
        try:
            # Evitar múltiples operaciones simultáneas
            if getattr(self, 'operacion_activa', False):
                logging.warning("Ya hay una operación activa, esperando a que termine antes de abrir otra.")
                return False
            if not self.verificar_limites_diarios():
                return False
            if time.time() - self.ultimo_tiempo_operacion < self.tiempo_espera:
                return False
            self.balance_pre_operacion = self.api.get_balance()
            if self.monto_actual > self.balance_pre_operacion:
                logging.error("Saldo insuficiente")
                self.resetear_estrategia()
                return False
            status, trade_id = self.api.buy(self.monto_actual, activo, direccion, 1)
            if status:
                logging.info(f"Operación {direccion.upper()} en {activo} - ${self.monto_actual:.2f}")
                self.operacion_activa = True
                self.id_operacion_actual = trade_id
                self.activo_actual = activo
                self.expiracion_operacion = time.time() + 60
                self.operaciones_hoy += 1
                self.ultimo_tiempo_operacion = time.time()
                self.registrar_operacion(activo, direccion, 'abierta')
                resultado = self.monitorear_operacion(activo, direccion)
                self.operacion_activa = False  # Liberar bandera tras finalizar
                return resultado
            else:
                logging.error("Error en operación")
                return False
        except Exception as e:
            logging.error(f"Error ejecutando operación: {str(e)}")
            self.operacion_activa = False
            return False

    def monitorear_operacion(self, activo, direccion):
        """Monitorea el resultado de una operación"""
        try:
            while time.time() < self.expiracion_operacion + 15:
                resultado, ganancia = self.verificar_resultado_operacion()
                
                if resultado in ['ganada', 'perdida']:
                    self.actualizar_estrategia(resultado, ganancia)
                    self.verificar_prediccion(resultado)
                    self.registrar_operacion(activo, direccion, resultado, ganancia)
                    self.ajustar_tiempo_espera(resultado)
                    return True
                    
                time.sleep(1)
                
            self.operacion_activa = False
            return False
            
        except Exception as e:
            logging.error(f"Error monitoreando operación: {str(e)}")
            return False

    def verificar_resultado_operacion(self):
        """Verifica el resultado de la operación actual"""
        if not self.operacion_activa:
            return None, None
            
        try:
            # Método 1: check_win_v4
            try:
                result = self.api.check_win_v4(self.id_operacion_actual)
                if result:
                    profit = result.get('win', 0)
                    return ('ganada', profit) if profit > 0 else ('perdida', -self.monto_actual)
            except:
                pass

            # Método 2: Por balance
            if time.time() > self.expiracion_operacion:
                current = self.api.get_balance()
                diff = current - self.balance_pre_operacion
                return ('ganada', diff) if diff > 0 else ('perdida', diff)
                
            return None, None
            
        except Exception as e:
            logging.error(f"Error verificando resultado: {str(e)}")
            return 'error', 0

    def verificar_limites_diarios(self):
        """Verifica los límites diarios de operaciones y riesgo"""
        if self.operaciones_hoy >= self.max_operaciones_diarias:
            logging.warning("Límite diario de operaciones alcanzado")
            return False
            
        current = self.api.get_balance()
        perdida = self.balance_inicial_dia - current
        ganancia = current - self.balance_inicial_dia
        
        if perdida >= (self.balance_inicial_dia * self.stop_loss_diario):
            logging.warning(f"Stop loss diario alcanzado. Pérdida: {perdida:.2f}")
            return False
            
        if ganancia >= (self.balance_inicial_dia * self.take_profit_diario):
            logging.warning(f"Take profit diario alcanzado. Ganancia: {ganancia:.2f}")
            return False
            
        return True

    def actualizar_estrategia(self, resultado, ganancia):
        """Actualiza la estrategia según el resultado de la operación"""
        self.balance = self.api.get_balance()
        
        if resultado == 'ganada':
            self.perdidas_consecutivas = 0
            self.monto_actual = self.monto_inicial
            logging.info(f"Ganada! Monto reiniciado a ${self.monto_inicial:.2f}")
        else:
            self.perdidas_consecutivas += 1
            
            if self.perdidas_consecutivas >= self.max_perdidas_consecutivas:
                self.resetear_estrategia()
            else:
                self.monto_actual *= self.multiplicador
                logging.warning(f"Pérdida #{self.perdidas_consecutivas}. Nuevo monto: ${self.monto_actual:.2f}")
        
        logging.info(f"Balance actual: ${self.balance:.2f}")

    def resetear_estrategia(self):
        """Reinicia la estrategia después de pérdidas consecutivas"""
        self.monto_actual = self.monto_inicial
        self.perdidas_consecutivas = 0
        logging.warning("Estrategia reiniciada. Esperando mejores condiciones...")
        time.sleep(60)  # Espera adicional después de reset

    def ajustar_tiempo_espera(self, resultado):
        """Ajusta el tiempo de espera entre operaciones"""
        self.tiempo_espera = random.randint(20, 40) if resultado == 'ganada' else random.randint(30, 60)
        logging.info(f"Próxima operación en {self.tiempo_espera}s")

    def verificar_prediccion(self, resultado):
        """Verifica si la última predicción fue correcta"""
        if not self.ultima_prediccion:
            return
            
        prediccion_correcta = (
            (resultado == 'ganada' and self.ultima_prediccion['direccion'] == 'call') or
            (resultado == 'perdida' and self.ultima_prediccion['direccion'] == 'put')
        )
        
        logging.info(f"Predicción: {self.ultima_prediccion['direccion'].upper()} "
                    f"(Prob: {self.ultima_prediccion['probabilidad']:.2%}) | "
                    f"Resultado: {resultado.upper()} "
                    f"{'✓' if prediccion_correcta else '✗'}")

    def registrar_operacion(self, activo, direccion, estado, ganancia=0):
        """Registra una operación en el historial"""
        operacion = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'activo': activo,
            'direccion': direccion,
            'monto': self.monto_actual,
            'estado': estado,
            'ganancia': ganancia,
            'balance': self.api.get_balance(),
            'prediccion': self.ultima_prediccion
        }
        
        self.historial_operaciones.append(operacion)
        self.guardar_historial()

    def guardar_historial(self):
        """Guarda el historial de operaciones en un archivo JSON"""
        try:
            hoy = datetime.now().strftime("%Y%m%d")
            archivo = self.directorio_historial / f"historial_{hoy}.json"
            
            with open(archivo, 'w', encoding='utf-8') as f:
                json.dump(self.historial_operaciones, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error guardando historial: {str(e)}")

    def mostrar_resumen(self):
        """Muestra un resumen de las operaciones del día"""
        if not self.historial_operaciones:
            return
            
        df = pd.DataFrame(self.historial_operaciones)
        operaciones = df[df['estado'].isin(['ganada', 'perdida'])]
        
        if len(operaciones) == 0:
            return
            
        ganadas = operaciones[operaciones['estado'] == 'ganada']
        perdidas = operaciones[operaciones['estado'] == 'perdida']
        
        logging.info("\n=== RESUMEN DEL DÍA ===")
        logging.info(f"Operaciones: {len(operaciones)}")
        logging.info(f"Ganadas: {len(ganadas)} ({len(ganadas)/len(operaciones):.1%})")
        logging.info(f"Perdidas: {len(perdidas)} ({len(perdidas)/len(operaciones):.1%})")
        logging.info(f"Balance final: ${self.balance:.2f}")
        logging.info("=======================")

    def ejecutar(self):
        """Método principal para ejecutar el bot"""
        try:
            if not self.conectar_api():
                return
                
            # Ejecutar backtesting inicial
            if not self.ejecutar_backtesting_activos():
                logging.error("Backtesting fallido. Revisar datos.")
                return
                
            logging.info("Bot listo para operar")
            
            while self.ejecutando:
                try:
                    # Seleccionar mejor activo
                    activo, condiciones = self.seleccionar_mejor_activo()
                    
                    if activo and condiciones:
                        # Actualizar datos del activo seleccionado
                        if not self.obtener_datos_historicos(activo, 300):
                            time.sleep(20)
                            continue
                            
                        # Verificar probabilidad mínima
                        umbral = 0.50  # Umbral más bajo que antes
                        if condiciones['probabilidad'] >= umbral:
                            self.ultima_prediccion = condiciones
                            self.ejecutar_operacion(activo, condiciones['direccion'])
                        else:
                            logging.info(f"Señal débil para {activo}. Prob: {condiciones['probabilidad']:.2%} < {umbral:.2%}")
                    
                    # Rotación de modelos cada 2 horas
                    if time.time() - getattr(self, 'ultimo_entrenamiento', 0) > 7200:
                        self.ejecutar_backtesting_activos()
                        self.ultimo_entrenamiento = time.time()
                    
                    time.sleep(15)
                    
                except KeyboardInterrupt:
                    self.ejecutando = False
                    logging.info("Detención solicitada por usuario")
                    
                except Exception as e:
                    logging.error(f"Error en bucle principal: {str(e)}")
                    time.sleep(30)
                    
        finally:
            self.detener()

    def detener(self):
        """Detiene el bot de manera controlada"""
        logging.info("Deteniendo bot...")
        
        if self.operacion_activa:
            resultado, _ = self.verificar_resultado_operacion()
            if resultado:
                self.actualizar_estrategia(resultado, 0)
                
        self.guardar_historial()
        self.mostrar_resumen()
        logging.info("Bot detenido correctamente")

# ----------------------------
# Ejecución Principal
# ----------------------------

if __name__ == "__main__":
    # Cargar credenciales desde variables de entorno
    load_dotenv()
    email = "rickybeto84@gmail.com"
    password = "IsabellaThiago123"
    
    if not email or not password:
        logging.error("Faltan credenciales. Configurar IQ_EMAIL e IQ_PASSWORD en .env")
        exit(1)
    
    try:
        # Configurar activos principales (no OTC)
        activos_principales = ["EURUSD-OTC", "GBPUSD-OTC", "EURGBP-OTC"]
        
        # Iniciar bot con gestión de riesgo conservadora
        bot = BotIQTrading(
            email=email,
            password=password,
            activos=activos_principales,
            tipo_cuenta="PRACTICE"  # Cambiar a "REAL" para cuenta real
        )
        
        # Configuración de trading conservadora
        bot.monto_inicial = 1.0  # Monto inicial pequeño
        bot.max_perdidas_consecutivas = 30  # Solo 3 pérdidas consecutivas permitidas
        bot.multiplicador = 3.0  # Multiplicador más conservador
        bot.stop_loss_diario = 1.0  # 10% de pérdida diaria máxima
        bot.take_profit_diario = 1.0  # 15% de ganancia diaria máxima
        
        # Ejecutar bot
        bot.ejecutar()
        
    except Exception as e:
        logging.critical(f"Error fatal: {str(e)}")
        exit(1)
