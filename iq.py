import flet as ft
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sqlite3
from iqoptionapi.stable_api import IQ_Option
import threading
import websocket

class IQOptionBot:
    def __init__(self, page):
        self.page = page
        self.api = None
        self.connected = False
        self.running = False
        self.email = ""
        self.password = ""
        self.current_prediction = ""
        self.current_confidence = 0
        self.trade_history = []
        self.balance = 1000  # Balance inicial simulado
        self.active_trades = []
        self.lock = threading.Lock()  # Añade un lock para thread safety
        
        # Configuración inicial
        self.pair = "EURUSD"
        self.candle_interval = 30  # 30 segundos
        self.min_confidence = 0.6
        # self.adx_threshold = 25
        self.initial_amount = 1
        self.martingale_multiplier = 3
        self.max_martingale_steps = 50
        
        # Componentes de la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de usuario con Flet"""
        self.email_field = ft.TextField(label="Email", width=300)
        self.password_field = ft.TextField(label="Password", password=True, width=300)
        self.start_button = ft.ElevatedButton("Iniciar Bot", on_click=self.start_bot_click)
        self.stop_button = ft.ElevatedButton("Detener Bot", on_click=self.stop_bot, disabled=True)
        self.status_text = ft.Text("Estado: Desconectado", color="red")
        self.balance_text = ft.Text(f"Balance: ${self.balance:.2f}")

        # Campo para intervalo de reentrenamiento (minutos)
        self.retrain_interval_field = ft.TextField(label="Reentrenar modelo cada (min)", width=200, value="60")

        # Campo para código 2FA (inicialmente oculto)
        self.twofa_field = ft.TextField(label="Código 2FA", width=200, visible=False)
        self.twofa_button = ft.ElevatedButton("Enviar 2FA", on_click=self.send_2fa_code, visible=False)
        self.twofa_dialog = None
        
        # Sección de predicción
        self.prediction_text = ft.Text("Predicción: --")
        self.confidence_text = ft.Text("Confianza: --%")
        
        # Historial de operaciones
        self.trade_history_view = ft.ListView(expand=True)
        
        # Operaciones activas
        self.active_trades_view = ft.ListView(expand=True, height=150)
        
        # Gráficos (placeholder)
        self.chart_placeholder = ft.Container(
            content=ft.Text("Gráfico de rendimiento aparecerá aquí"),
            border=ft.border.all(1),
            padding=10,
            height=200
        )
        
        # Diseño principal
        self.page.add(
            ft.Column([
                ft.Row([
                    self.email_field,
                    self.password_field,
                    self.retrain_interval_field  # Ahora visible junto a email y password
                ]),
                ft.Row([self.start_button, self.stop_button, self.status_text, self.balance_text]),
                ft.Row([self.twofa_field, self.twofa_button]),
                ft.Divider(),
                ft.Row([
                    ft.Column([
                        ft.Text("Información de Predicción:", weight=ft.FontWeight.BOLD),
                        self.prediction_text,
                        self.confidence_text,
                        ft.Text("Operaciones Activas:", weight=ft.FontWeight.BOLD),
                        self.active_trades_view
                    ], width=300),
                    ft.VerticalDivider(),
                    ft.Column([
                        ft.Text("Historial de Operaciones:", weight=ft.FontWeight.BOLD),
                        self.trade_history_view
                    ], expand=True)
                ]),
                ft.Divider(),
                self.chart_placeholder
            ])
        )
    
    def start_bot_click(self, e):
        """Maneja el evento de clic en el botón de inicio"""
        self.email = self.email_field.value
        self.password = self.password_field.value
        
        if not self.email or not self.password:
            self.show_alert("Error", "Por favor ingresa email y contraseña")
            return
        
        self.start_button.disabled = True
        self.stop_button.disabled = False
        self.page.update()
        
        # Iniciar el bot en un hilo separado
        threading.Thread(target=self.run_bot, daemon=True).start()
    
    def stop_bot(self, e):
        """Detiene la ejecución del bot"""
        self.running = False
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.status_text.value = "Estado: Deteniendo..."
        self.status_text.color = "orange"
        self.page.update()
    
    def connect_iqoption(self, max_retries=3):
        """Conecta a la API de IQ Option con reintentos y soporte para 2FA"""
        self.status_text.value = "Estado: Conectando..."
        self.status_text.color = "orange"
        self.page.update()

        for attempt in range(max_retries):
            try:
                servers = ['iqoption.com', 'iqoption.com.br', 'iqoption.eu']
                self.api = IQ_Option(self.email, self.password)
                connected = False
                for server in servers:
                    try:
                        self.api.set_server(server)
                        if self.api.connect():
                            connected = True
                            break
                    except Exception:
                        continue
                if not connected:
                    self.update_log(f"Intento {attempt + 1}: No se pudo conectar a ningún servidor")
                    time.sleep(2)
                    continue

                # Verificar autenticación de dos factores si es necesario
                if self.api.check_connect() == '2FA':
                    self.update_log("Se requiere autenticación de dos factores (2FA)")
                    self.status_text.value = "Estado: Esperando 2FA..."
                    self.status_text.color = "orange"
                    self.twofa_field.visible = True
                    self.twofa_button.visible = True
                    self.page.update()
                    self.waiting_2fa = True
                    return False

                time.sleep(2)
                if self.api.check_connect():
                    self.api.change_balance("PRACTICE")
                    self.connected = True
                    self.status_text.value = "Estado: Conectado"
                    self.status_text.color = "green"
                    self.running = True
                    self.page.update()
                    return True
            except websocket._exceptions.WebSocketConnectionClosedException:
                self.update_log(f"Intento {attempt + 1}: Conexión cerrada inesperadamente, reintentando...")
                time.sleep(3)
            except Exception as e:
                self.update_log(f"Intento {attempt + 1}: Error inesperado - {str(e)}")
                time.sleep(3)
        self.status_text.value = "Estado: Error de conexión"
        self.status_text.color = "red"
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.page.update()
        return False

    def send_2fa_code(self, e):
        """Envía el código 2FA a la API y reintenta la conexión"""
        code = self.twofa_field.value.strip()
        if not code:
            self.show_alert("Error", "Por favor ingresa el código 2FA")
            return
        try:
            self.api.two_factor(code)
            time.sleep(2)
            if self.api.check_connect():
                self.api.change_balance("PRACTICE")
                self.connected = True
                self.status_text.value = "Estado: Conectado"
                self.status_text.color = "green"
                self.running = True
                self.twofa_field.visible = False
                self.twofa_button.visible = False
                self.page.update()
                self.update_log("Autenticación 2FA exitosa. Bot conectado.")
            else:
                self.update_log("Código 2FA incorrecto o expirado.")
        except Exception as ex:
            self.update_log(f"Error al enviar 2FA: {str(ex)}")
        self.page.update()
    
    def get_balance_safe(api, max_retries=3):
        for attempt in range(max_retries):
            try:
                balance = api.get_balance()
                return balance
            except websocket._exceptions.WebSocketConnectionClosedException:
                print(f"Intento {attempt + 1}: Conexión cerrada al obtener balance, reconectando...")
                api.reconnect()
                time.sleep(2)
            except Exception as e:
                print(f"Intento {attempt + 1}: Error al obtener balance - {str(e)}")
                time.sleep(1)
        
        raise ConnectionError("No se pudo obtener el balance después de varios intentos")


    
    def is_high_impact_news(self):
        """Verifica si hay noticias importantes (simulación)"""
        now = datetime.now()
        # Horarios de noticias importantes (ejemplo)
        news_times = [
            (now.replace(hour=13, minute=30, second=0), now.replace(hour=14, minute=0, second=0)),  # NFP
            (now.replace(hour=14, minute=0, second=0), now.replace(hour=14, minute=30, second=0))   # FED
        ]
        for start, end in news_times:
            if start <= now <= end:
                return True
        return False
    
    def predict_direction(self, df):
        """Predice la dirección usando Random Forest"""
        try:
            model = joblib.load('option_predictor.joblib')
        except:
            self.update_log("Modelo no encontrado. Entrenando nuevo modelo...")
            model = self.train_model(df)
        
        features = ['open', 'high', 'low', 'close', 'volume']
        X = df[features].iloc[-1:].values
        proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        confidence = max(proba)
        
        return prediction, confidence
    
    def train_model(self, df):
        """Entrena el modelo de Random Forest con validación cruzada"""
        from sklearn.model_selection import cross_val_score
        features = ['open', 'high', 'low', 'close', 'volume']
        target = 'direction'
        X = df[features]
        y = df[target]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Validación cruzada 5-fold
        try:
            cv_scores = cross_val_score(model, X, y, cv=5)
            mean_cv = np.mean(cv_scores)
            self.update_log(f"Precisión media (cross-val 5-fold): {mean_cv:.2%}")
        except Exception as ex:
            self.update_log(f"Error en validación cruzada: {str(ex)}")
        # Entrenamiento final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.update_log(f"Precisión en test: {accuracy:.2%}")
        joblib.dump(model, 'option_predictor.joblib')
        return model
    
    def execute_trade(self, direction, amount):
        """Ejecuta una operación en IQ Option"""
        try:
            _, trade_id = self.api.buy(amount, self.pair, direction, self.candle_interval)
            self.active_trades.append({
                'id': trade_id,
                'pair': self.pair,
                'direction': direction,
                'amount': amount,
                'time': datetime.now()
            })
            self.update_active_trades()
            return trade_id
        except Exception as e:
            self.update_log(f"Error al ejecutar operación: {str(e)}")
            return None
    
    def check_trade_result(self, trade_id):
        """Verifica el resultado de una operación"""
        try:
            result = self.api.check_win_v2(trade_id)
            if result > 0:
                return 'win'
            elif result < 0:
                return 'loss'
            else:
                return 'even'
        except:
            return 'unknown'
    
    def update_ui_predictions(self, prediction, confidence):
        """Actualiza la UI con la última predicción"""
        self.current_prediction = prediction
        self.current_confidence = confidence
        
        self.prediction_text.value = f"Predicción: {prediction}"
        self.confidence_text.value = f"Confianza: {confidence:.2%}"
        
        if confidence >= self.min_confidence:
            self.prediction_text.color = "green"
        else:
            self.prediction_text.color = "red"
        
        self.page.update()
    
    def update_log(self, message):
        """Agrega un mensaje al historial"""
        now = datetime.now().strftime("%H:%M:%S")
        self.trade_history_view.controls.append(ft.Text(f"[{now}] {message}"))
        if len(self.trade_history_view.controls) > 100:
            self.trade_history_view.controls.pop(0)
        self.page.update()
    
    def update_active_trades(self):
        """Actualiza la lista de operaciones activas"""
        self.active_trades_view.controls.clear()
        for trade in self.active_trades:
            self.active_trades_view.controls.append(
                ft.Text(f"{trade['time'].strftime('%H:%M:%S')} - {trade['pair']} {trade['direction']} ${trade['amount']:.2f}")
            )
        self.page.update()
    
    def update_balance(self, change):
        """Actualiza el balance simulado"""
        self.balance += change
        self.balance_text.value = f"Balance: ${self.balance:.2f}"
        self.page.update()
    
    def show_alert(self, title, message):
        """Muestra una alerta al usuario"""
        self.page.dialog = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Text(message),
            on_dismiss=lambda e: print("Dialog dismissed!")
        )
        self.page.dialog.open = True
        self.page.update()
    
    def run_bot(self):
        """Lógica principal del bot con reentrenamiento periódico configurable"""
        if not self.connect_iqoption():
            return

        martingale = MartingaleManager(
            initial_amount=self.initial_amount,
            multiplier=self.martingale_multiplier,
            max_steps=self.max_martingale_steps
        )

        self.update_log("Bot iniciado correctamente")

        # Reentrenamiento periódico
        last_retrain = time.time()
        try:
            retrain_interval_min = float(self.retrain_interval_field.value)
        except Exception:
            retrain_interval_min = 60.0
        retrain_interval_sec = retrain_interval_min * 60

        while self.running:
            try:
                # Obtener datos del mercado
                candles = self.get_candles(1000)
                candles = self.calculate_adx(candles)

                # Reentrenar modelo si ha pasado el intervalo
                if time.time() - last_retrain > retrain_interval_sec:
                    self.update_log("Reentrenando modelo por intervalo configurado...")
                    self.train_model(candles)
                    last_retrain = time.time()

                # Hacer predicción
                prediction, confidence = self.predict_direction(candles)
                current_trend = candles['trend'].iloc[-1]

                # Actualizar UI
                self.update_ui_predictions(prediction, confidence)

                # Verificar condiciones para operar
                if (confidence >= self.min_confidence and
                    ((prediction == 'CALL' and current_trend == 'UP') or 
                     (prediction == 'PUT' and current_trend == 'DOWN')) and
                    not self.is_high_impact_news()):

                    # Calcular monto con martingala
                    amount = martingale.current_amount

                    # Verificar balance suficiente
                    if amount > self.balance:
                        self.update_log("Balance insuficiente para operar")
                        martingale.next_bet(True)  # Reset martingale
                        continue

                    # Ejecutar operación
                    self.update_log(f"Operando {prediction} con ${amount:.2f} (Confianza: {confidence:.2%})")
                    trade_id = self.execute_trade(prediction, amount)

                    if trade_id:
                        # Esperar resultado
                        time.sleep(self.candle_interval + 2)  # Esperar vela + margen

                        # Verificar resultado
                        result = self.check_trade_result(trade_id)
                        self.update_log(f"Resultado: {result.upper()}")

                        # Actualizar balance (simulado)
                        if result == 'win':
                            self.update_balance(amount * 0.8)  # IQ Option paga ~80%
                        else:
                            self.update_balance(-amount)

                        # Actualizar martingala
                        martingale.next_bet(result == 'win')

                        # Eliminar operación de activas
                        self.active_trades = [t for t in self.active_trades if t['id'] != trade_id]
                        self.update_active_trades()

                # Esperar para la próxima iteración
                time.sleep(5)

            except Exception as e:
                self.update_log(f"Error en ciclo principal: {str(e)}")
                time.sleep(10)

        self.status_text.value = "Estado: Detenido"
        self.status_text.color = "red"
        self.page.update()
        self.update_log("Bot detenido")

class MartingaleManager:
    """Gestiona la estrategia de Martingala"""
    def __init__(self, initial_amount=1, multiplier=3, max_steps=5):
        self.initial_amount = initial_amount
        self.multiplier = multiplier
        self.max_steps = max_steps
        self.current_step = 0
        self.current_amount = initial_amount
        
    def next_bet(self, win):
        """Calcula el próximo monto basado en el resultado anterior"""
        if win:
            self.current_step = 0
            self.current_amount = self.initial_amount
        else:
            self.current_step += 1
            if self.current_step <= self.max_steps:
                self.current_amount *= self.multiplier
            else:
                self.current_amount = self.initial_amount
                self.current_step = 0
        return self.current_amount

import threading
import time
import logging
from prueba import BotIQTrading

import flet as ft

class BotFletApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "RB BOT OPCIONES BINARIAS"
        self.page.window_width = 900
        self.page.window_height = 700
        # Fondo gris medio oscuro, botones y tarjetas en grises sutiles
        self.page.bgcolor = "#B0B3B8"  # Gris medio oscuro
        self.bot_thread = None
        self.bot = None
        self.running = False
        # Reduce logs area, maximize chart area
        self.logs = ft.ListView(height=80, spacing=2, auto_scroll=False)
        # Gráficos: tamaño grande, cada uno en su columna
        self.chart_img = ft.Image(src="", width=420, height=260, fit=ft.ImageFit.CONTAIN, visible=False)
        self.bar_img = ft.Image(src="", width=420, height=260, fit=ft.ImageFit.CONTAIN, visible=False)
        self.tarjetas_title = ft.Text("PERDIDAS Y GANANCIAS", size=18, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
        # ISA 101: Colores estándar
        isa_text = "#111"  # Gris muy oscuro para texto principal
        card_gray = "#F2F3F4"  # Gris claro para tarjetas
        button_gray = "#888A8D"  # Gris medio para botones
        button_gray_hover = "#6D6F73"  # Gris más oscuro para hover
        button_gray_disabled = "#D1D3D4"  # Gris claro para deshabilitado
        # Make PYG/%PYG cards larger, font size 75% of previous (was 22, now ~16.5)
        self.percent_card = ft.Container(
            content=ft.Text("% PYG: --", size=16, weight=ft.FontWeight.BOLD, color=isa_text),
            bgcolor=card_gray,
            border_radius=14,
            padding=12,
            alignment=ft.alignment.center,
            width=160,
            height=54,
            visible=False
        )
        self.pyg_usd_card = ft.Container(
            content=ft.Text("PYG: -- USD", size=16, weight=ft.FontWeight.BOLD, color=isa_text),
            bgcolor=card_gray,
            border_radius=14,
            padding=12,
            alignment=ft.alignment.center,
            width=160,
            height=54,
            visible=False
        )
        self.email_input = ft.TextField(label="Email", width=350, bgcolor="#FFF", color=ft.Colors.GREY_900)
        self.password_input = ft.TextField(label="Password", password=True, width=350, bgcolor="#FFF", color=ft.Colors.GREY_900)
        self.retrain_interval_field = ft.TextField(label="Reentrenar modelo cada (min)", width=180, value="60", bgcolor="#FFF", color=ft.Colors.GREY_900)
        self.account_type = ft.Dropdown(
            label="Tipo de cuenta",
            width=180,
            value="PRACTICE",
            options=[
                ft.dropdown.Option("REAL", "Real"),
                ft.dropdown.Option("PRACTICE", "Practice"),
                ft.dropdown.Option("TOURNAMENT", "Tournament"),
            ]
        )
        self.start_btn = ft.ElevatedButton("Iniciar Bot", on_click=self.start_bot, width=150, bgcolor=button_gray, color=ft.Colors.BLACK)
        self.stop_btn = ft.ElevatedButton("Detener Bot", on_click=self.stop_bot, width=150, disabled=True, bgcolor=button_gray, color=ft.Colors.BLACK)
        self.status_text = ft.Text("Estado: Desconectado", color=ft.Colors.GREY_900)
        self.balance_text = ft.Text("Balance: --", color=ft.Colors.GREY_900)
        self.balance_inicial_text = ft.Text("Balance inicial: --", color=ft.Colors.GREY_900)

        # NUEVO: Solo los títulos, sin mostrar valores
        self.credenciales_text = ft.Text("Credenciales", color=ft.Colors.GREY_900, size=16, weight=ft.FontWeight.BOLD)
        self.estado_bot_text = ft.Text("Estado del Bot", color=ft.Colors.GREY_900, size=16, weight=ft.FontWeight.BOLD)

        # Área de rendimiento en tres columnas: balance | barras | tarjetas
        self.page.add(
            ft.Column([
                ft.Row([
                    ft.Text("RB BOT OPCIONES BINARIAS", size=24, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Container(self.credenciales_text, alignment=ft.alignment.center, padding=5),
                ft.Row([
                    self.email_input,
                    self.password_input,
                    self.account_type,
                    self.retrain_interval_field
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Container(self.estado_bot_text, alignment=ft.alignment.center, padding=5),
                ft.Row([
                    self.start_btn, self.stop_btn, self.status_text, self.balance_text, self.balance_inicial_text
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Row([
                    ft.Text("Rentabilidad del Bot", size=22, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(
                    ft.Row([
                        ft.Column([
                            self.chart_img
                        ], alignment=ft.MainAxisAlignment.CENTER, expand=True),
                        ft.VerticalDivider(width=1),
                        ft.Column([
                            self.bar_img
                        ], alignment=ft.MainAxisAlignment.CENTER, expand=True),
                        ft.VerticalDivider(width=1),
                        ft.Column([
                            self.tarjetas_title,
                            ft.Container(height=18),
                            self.percent_card,
                            ft.Container(height=18),
                            self.pyg_usd_card
                        ], alignment=ft.MainAxisAlignment.CENTER, expand=True)
                    ], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.START, expand=True),
                    padding=10,
                    expand=True,
                    bgcolor=None
                ),
                ft.Divider(),
                ft.Row([
                    ft.Text("Logs", size=18, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER),
                self.logs
            ], expand=True)
        )
    def mostrar_grafico_rentabilidad(self):
        """Genera y muestra los gráficos de rentabilidad y barras con Plotly, y la tarjeta de % ganancia/perdida de la última operación."""
        try:
            if not self.running:
                return
            import plotly.graph_objects as go
            import io, base64
            import pandas as pd
            # Mostrar aunque haya solo 1 operación cerrada
            if not self.bot or not hasattr(self.bot, 'historial_operaciones'):
                self.chart_img.visible = False
                self.bar_img.visible = False
                self.percent_card.visible = False
                self.pyg_usd_card.visible = False
                self.page.update()
                return
            df = pd.DataFrame(self.bot.historial_operaciones)
            # Asegurarse de que haya al menos una operación y columna balance
            if 'balance' not in df.columns or df['balance'].count() < 1:
                self.chart_img.visible = False
                self.bar_img.visible = False
                self.percent_card.visible = False
                self.pyg_usd_card.visible = False
                self.page.update()
                return

            # Para la primera operación, graficar aunque solo haya una
            df_balance = df[df['estado'].isin(['ganada', 'perdida'])].copy()
            if df_balance.shape[0] == 0 and df.shape[0] >= 1:
                # Si no hay cerradas, graficar la primera
                df_balance = df.iloc[[0]].copy()

            # --- GRAFICO DE BALANCE ---
            if df_balance.shape[0] >= 1:
                idx_primera = df.index.get_loc(df_balance.index[0])
                if idx_primera == 0:
                    balance_inicial = df['balance'].iloc[0]
                else:
                    balance_inicial = df['balance'].iloc[idx_primera-1]
                balance_final = df_balance['balance'].iloc[-1]
                line_color = '#00FF7F' if balance_final >= balance_inicial else '#FF3B30'
                marker_color = line_color
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_balance['timestamp'],
                    y=df_balance['balance'],
                    mode='lines+markers',
                    name='Balance',
                    line=dict(color=line_color, width=3),
                    marker=dict(color=marker_color, size=6, line=dict(width=1, color='#222')),
                    hovertemplate='<b>Hora:</b> %{x}<br><b>Balance:</b> %{y:.2f} $<extra></extra>'
                ))
                fig.update_layout(
                    title=dict(text='<b>Evolución del Balance</b>', font=dict(size=32, color='#0099FF'), x=0.5),
                    xaxis=dict(
                        title=dict(text='Hora', font=dict(size=24, color='black')),
                        showgrid=False, zeroline=False, showline=True, linecolor='#888', tickfont=dict(size=22, color='black')
                    ),
                    yaxis=dict(
                        title=dict(text='Balance ($)', font=dict(size=24, color='black')),
                        showgrid=False, zeroline=False, showline=True, linecolor='#888', tickfont=dict(size=22, color='black')
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='black', size=24),
                    height=260,
                    margin=dict(l=30, r=30, t=80, b=30),
                    showlegend=False,
                    hovermode='x unified'
                )
                buf = io.BytesIO()
                fig.write_image(buf, format='png', scale=2)
                buf.seek(0)
                self.chart_img.src_base64 = base64.b64encode(buf.read()).decode('utf-8')
                self.chart_img.visible = True
            else:
                self.chart_img.visible = False

            # --- GRAFICO DE BARRAS ---
            df_ops = df[df['estado'].isin(['ganada', 'perdida'])]
            if df_ops.shape[0] == 0 and df.shape[0] >= 1:
                win_count = 0
                loss_count = 0
                bar_fig = go.Figure(data=[
                    go.Bar(
                        name='Ganadas',
                        x=['Ganadas'],
                        y=[win_count],
                        marker_color='#00FF7F',
                        width=0.5,
                        hovertemplate='<b>Ganadas:</b> %{y}<extra></extra>'
                    ),
                    go.Bar(
                        name='Perdidas',
                        x=['Perdidas'],
                        y=[loss_count],
                        marker_color='#FF3B30',
                        width=0.5,
                        hovertemplate='<b>Perdidas:</b> %{y}<extra></extra>'
                    )
                ])
            else:
                win_count = (df_ops['estado'] == 'ganada').sum()
                loss_count = (df_ops['estado'] == 'perdida').sum()
                bar_fig = go.Figure(data=[
                    go.Bar(
                        name='Ganadas',
                        x=['Ganadas'],
                        y=[win_count],
                        marker_color='#00FF7F',
                        width=0.5,
                        hovertemplate='<b>Ganadas:</b> %{y}<extra></extra>'
                    ),
                    go.Bar(
                        name='Perdidas',
                        x=['Perdidas'],
                        y=[loss_count],
                        marker_color='#FF3B30',
                        width=0.5,
                        hovertemplate='<b>Perdidas:</b> %{y}<extra></extra>'
                    )
                ])
            bar_fig.update_layout(
                title=dict(text='<b>Ganadas vs Perdidas</b>', font=dict(size=32, color='#0099FF'), x=0.5),
                xaxis=dict(
                    title=dict(text='', font=dict(size=24, color='black')),
                    showgrid=False, zeroline=False, showline=True, linecolor='#888', tickfont=dict(size=22, color='black')
                ),
                yaxis=dict(
                    title=dict(text='', font=dict(size=24, color='black')),
                    showgrid=False, zeroline=False, showline=True, linecolor='#888', tickfont=dict(size=22, color='black')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black', size=24),
                barmode='group',
                height=260,
                margin=dict(l=30, r=30, t=80, b=30),
                showlegend=False,
                hovermode='x unified'
            )
            buf2 = io.BytesIO()
            bar_fig.write_image(buf2, format='png', scale=2)
            buf2.seek(0)
            self.bar_img.src_base64 = base64.b64encode(buf2.read()).decode('utf-8')
            self.bar_img.visible = True

            # Tarjeta de porcentaje de ganancia acumulada (% PYG) y tarjeta de ganancia/pérdida en USD (PYG)
            # El balance inicial debe ser el balance antes de la PRIMERA operación cerrada (o la primera operación si no hay cerradas)
            if df_balance.shape[0] >= 1:
                idx_primera = df.index.get_loc(df_balance.index[0])
                if idx_primera == 0:
                    balance_inicial = df['balance'].iloc[0]
                else:
                    balance_inicial = df['balance'].iloc[idx_primera-1]
                balance_actual = df_balance['balance'].iloc[-1]
                usd = balance_actual - balance_inicial
                # Si el balance inicial es 0, mostrar 0% y 0 USD
                if balance_inicial == 0:
                    pct = 0.0
                else:
                    pct = ((balance_actual - balance_inicial) / balance_inicial) * 100
                color = "#00FF7F" if pct >= 0 else "#FF3B30"
                # Font size 75% of previous (was 22, now 16)
                self.percent_card.content = ft.Text(f"% PYG: {pct:.2f}%", size=16, weight=ft.FontWeight.BOLD, color=color)
                self.percent_card.bgcolor = ft.Colors.WHITE if pct >= 0 else ft.Colors.RED_100
                self.percent_card.visible = True
                usd_color = "#00FF7F" if usd >= 0 else "#FF3B30"
                self.pyg_usd_card.content = ft.Text(f"PYG: {usd:+.2f} USD", size=16, weight=ft.FontWeight.BOLD, color=usd_color)
                self.pyg_usd_card.bgcolor = ft.Colors.WHITE if usd >= 0 else ft.Colors.RED_100
                self.pyg_usd_card.visible = True
                # Mostrar el balance inicial de la operación
                self.balance_inicial_text.value = f"Balance inicial: ${balance_inicial:.2f}"
                self.balance_inicial_text.visible = True
            else:
                self.percent_card.visible = False
                self.pyg_usd_card.visible = False
                self.balance_inicial_text.value = "Balance inicial: --"
                self.balance_inicial_text.visible = True
            self.page.update()
        except RuntimeError as ex:
            if 'cannot schedule new futures after shutdown' in str(ex):
                self.log('Kaleido se cerró. No se pueden generar más gráficos hasta reiniciar la app.', color='red')
                return
            else:
                self.log(f'Error al generar gráficos: {str(ex)}', color='red')
                self.chart_img.visible = False
                self.bar_img.visible = False
                self.percent_card.visible = False
                self.pyg_usd_card.visible = False
                self.page.update()
        except Exception as ex:
            # Captura cualquier error de Kaleido/Choreo y lo muestra en logs, pero no detiene la app
            self.log(f"Error al generar gráficos: {str(ex)}", color="red")
            self.chart_img.visible = False
            self.bar_img.visible = False
            self.percent_card.visible = False
            self.pyg_usd_card.visible = False
            self.page.update()

    def log(self, msg, color=None):
        # Cambiar colores: naranja/amarillo a negro, verde a azul eléctrico
        color_map = {
            "orange": ft.Colors.BLACK,
            "yellow": ft.Colors.BLACK,
            "#FFA500": ft.Colors.BLACK,
            "#FFD700": ft.Colors.BLACK,
            "green": "#0099FF",
            "#00FF00": "#0099FF",
            "#00FF7F": "#0099FF",
        }
        color_final = color_map.get(color, color)
        self.logs.controls.insert(0, ft.Text(msg, color=color_final))
        if len(self.logs.controls) > 15:
            self.logs.controls = self.logs.controls[:15]
        self.mostrar_grafico_rentabilidad()
        self.page.update()

    def start_bot(self, e):
        email = self.email_input.value
        password = self.password_input.value
        account_type = self.account_type.value
        if not email or not password:
            self.log("Por favor ingresa email y contraseña", color="red")
            return
        self.page.update()
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.status_text.value = "Estado: Conectando..."
        self.status_text.color = "orange"
        self.page.update()
        self.running = True
        self.bot_thread = threading.Thread(target=self.run_bot, args=(email, password, account_type), daemon=True)
        self.bot_thread.start()

    def stop_bot(self, e):
        self.running = False
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.status_text.value = "Estado: Detenido"
        self.status_text.color = "red"
        self.page.update()
        if self.bot:
            try:
                self.bot.ejecutando = False
            except Exception:
                pass
    def run_bot(self, email, password, account_type):
        try:
            self.bot = BotIQTrading(email=email, password=password, account_type=account_type)
            self.log("Conectando a IQ Option...", color="orange")
            self.page.update()
            if not self.bot.conectar_api():
                self.status_text.value = "Estado: Error de conexión"
                self.status_text.color = "red"
                self.page.update()
                self.log("No se pudo conectar a IQ Option", color="red")
                self.start_btn.disabled = False
                self.stop_btn.disabled = True
                self.page.update()
                return
            self.status_text.value = "Estado: Conectado"
            self.status_text.color = "green"
            self.page.update()
            self.balance_text.value = f"Balance: ${self.bot.balance:.2f}"
            self.page.update()
            self.log("Bot conectado. Ejecutando estrategia...", color="green")
            # Ejecutar el bot principal en un hilo aparte para no bloquear la UI
            def bot_loop():
                try:
                    self.bot.ejecutando = True
                    self.log("Iniciando ciclo de trading...", color="orange")
                    self.bot.ejecutando = True
                    self.bot.conectado = True
                    self.page.update()
                    while self.running and self.bot.ejecutando:
                        try:
                            if not self.bot.modelos or not self.bot.datos_historicos:
                                self.log("Entrenando modelos y ejecutando backtesting...", color="orange")
                                ok = self.bot.ejecutar_backtesting_activos()
                                if not ok:
                                    self.log("No se pudo entrenar modelos ni hacer backtesting.", color="red")
                                    time.sleep(15)
                                    continue
                            activos = list(self.bot.modelos.keys()) if hasattr(self.bot, 'modelos') else []
                            mejores = []
                            for activo in activos:
                                ok_30s = self.bot.obtener_datos_historicos(activo, 300)
                                if not ok_30s:
                                    self.log(f"No se pudo obtener datos 30s para {activo}", color="red")
                                    continue
                                condiciones = None
                                if hasattr(self.bot, 'seleccionar_condiciones_por_activo'):
                                    condiciones = self.bot.seleccionar_condiciones_por_activo(activo)
                                if not condiciones:
                                    continue
                                mejores.append((activo, condiciones))
                            # Buscar el activo con mayor probabilidad
                            # Filtrar activos con probabilidad >= 60%
                            mejores = [x for x in mejores if x[1]['probabilidad'] >= 0.6]
                            if mejores:
                                activo_max, condiciones_max = max(mejores, key=lambda x: x[1]['probabilidad'])
                                self.bot.ultima_prediccion = condiciones_max
                                probabilidad = condiciones_max['probabilidad']
                                self.log(f"Intentando operar {activo_max} {condiciones_max['direccion']} (Probabilidad: {probabilidad:.2%})", color="orange")
                                opero = self.bot.ejecutar_operacion(activo_max, condiciones_max['direccion'])
                                try:
                                    self.balance_text.value = f"Balance: ${self.bot.api.get_balance():.2f}"
                                    self.page.update()
                                except Exception:
                                    pass
                                if opero:
                                    self.log(f"Operación ejecutada en {activo_max} {condiciones_max['direccion']} (Probabilidad: {probabilidad:.2%})", color="green")
                                else:
                                    self.log(f"No se pudo ejecutar operación en {activo_max} (Probabilidad: {probabilidad:.2%})", color="red")
                                # Forzar actualización del gráfico de balance después de cada operación
                                self.mostrar_grafico_rentabilidad()
                            else:
                                self.log("No se encontró activo óptimo para operar (ninguno supera 60% de probabilidad)", color="blue")
                            # Actualizar balance también después de cada ciclo (por si se cerró una operación)
                            try:
                                self.balance_text.value = f"Balance: ${self.bot.api.get_balance():.2f}"
                                self.page.update()
                            except Exception:
                                pass
                            # Forzar actualización del gráfico de balance después de cada ciclo
                            self.mostrar_grafico_rentabilidad()
                            time.sleep(15)
                        except Exception as e:
                            self.log(f"Error en ciclo de trading: {str(e)}", color="red")
                            time.sleep(20)
                    self.log("Bot detenido.", color="red")
                    self.page.update()
                except Exception as ex:
                    self.log(f"Error en ejecución del bot: {str(ex)}", color="red")
                    self.page.update()
                finally:
                    self.running = False
                    self.status_text.value = "Estado: Detenido"
                    self.status_text.color = "red"
                    self.start_btn.disabled = False
                    self.stop_btn.disabled = True
                    self.page.update()
            trading_thread = threading.Thread(target=bot_loop, daemon=True)
            trading_thread.start()
        except Exception as ex:
            self.log(f"Error: {str(ex)}", color="red")
            self.status_text.value = "Estado: Error"
            self.status_text.color = "red"
            self.page.update()
            self.page.update()


def main(page: ft.Page):
    BotFletApp(page)

if __name__ == "__main__":
    ft.app(target=main)
