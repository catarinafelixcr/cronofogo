import base64
from io import BytesIO, StringIO
from pathlib import Path 
from typing import Dict, List, Any, Union, Tuple, Optional

# Dash e Plotly
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context, callback 
from plotly.subplots import make_subplots

#  Visualizações Específicas 
from wordcloud import WordCloud 
from matplotlib import cm # colormaps 
import numpy as np
import matplotlib.colors as mcolors # conversões e manipulação de cores
import pandas as pd 
import re 

# constantes
BASE_DIR = Path(__file__).resolve().parent 
CSV_2012_2021 =  BASE_DIR / "data" / "incendios_2012a2021.csv"
CSV_2022 = BASE_DIR / "data" / "dados_2022.csv" 

LOGO_SRC = "/assets/logo.png"

PALETTE: Dict[str, str] = {
    "brand": "#ffb66d", 
    "brand_dark": "#a3302c", 
    "accent_red": "#E74C3C",
    "accent_orange": "#F39C12", 
    "bg": "#F8F9FA", 
    "font": "#2C3E50",
    "sidebar_text": "#FFFFFF", 
    "button_hover": "#A93226", 
    "card_bg": "#FFFFFF",
    "slider_handle": "#FFFFFF", 
    "slider_rail": "#E74C3C",
    "escala_area_inicio": "#FFF5E5",
    "escala_area_meio": "#FFB66D",  
    "escala_area_fim": "#A3302C",
    "prediction_year_color": "#FFA500" # para destacar 2022
}


FONT_SIZE_CHART_TITLE = 14
FONT_SIZE_AXIS_TITLE = 10
FONT_SIZE_TICK_LABEL = 9
FONT_SIZE_LEGEND_TITLE = 10
FONT_SIZE_LEGEND_ITEM = 9
FONT_SIZE_COLORBAR_TITLE = 10
FONT_SIZE_COLORBAR_TICK = 9
FONT_SIZE_MAP_TEXT = 9
FONT_SIZE_ANIMATION_VALUE = 12
FONT_SIZE_ANIMATION_BUTTON = 12
FONT_SIZE_PIE_TEXT = 9

_BASE_MESES_INICIAIS = {1: "J", 2: "F", 3: "M", 4: "A", 5: "M", 6: "Jn",
                                      7: "Jl", 8: "A", 9: "Set", 10: "O", 11: "N", 12: "D"}

MESES_CURTO_RADIO = { # botões de rádio da sidebar
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}

_BASE_MESES_EXTENSO = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

MESES_EXTENSO = {0: "Todos os Meses", **_BASE_MESES_EXTENSO}
MESES_CURTO_RADIO_INVERSO = {v: k for k, v in MESES_CURTO_RADIO.items()} # mapeamento inverso

# para variáveis meteorológicas (usado em dropdowns e labels)
METEO_VARS = [
    {"label": "Humidade", "value": "HUMIDADERELATIVA"},
    {"label": "Temperatura", "value": "TEMPERATURA"},
    {"label": "Vento", "value": "VENTOINTENSIDADE"}
]

METEO_LABELS_MAP = {item["value"]: item["label"] for item in METEO_VARS}

# Função para simplificar as famílias de causas dos incêndios
def simplificar_familia(causa):
    if pd.isna(causa):
        return "Desconhecida"
    causa = causa.lower()

    # agrupam causas semelhantes sob um nome comum
    causa = re.sub(r".*pasto.*", "Queimada - Pasto", causa)
    causa = re.sub(r".*sobrantes.*", "Queimada - Sobrantes", causa)
    causa = re.sub(r".*maquinaria.*", "Uso de Maquinaria", causa)
    causa = re.sub(r".*lazer.*", "Negligência em Lazer", causa)
    causa = re.sub(r".*gestão.*vegetação.*", "Gestão de Vegetação", causa)
    causa = re.sub(r"incêndios florestais", "Incêndio Florestal", causa)
    causa = re.sub(r".*desconhecida.*", "Desconhecida", causa)
    # causa = re.sub(r"[^a-zà-ú ]", "", causa)
    causa = causa.strip().title()
    return causa

# Carregamento
df_2012_2021 = pd.read_csv(CSV_2012_2021)
df_2022 = pd.read_csv(CSV_2022) 
DF = pd.concat([df_2012_2021, df_2022], ignore_index=True) 
DF.columns = DF.columns.str.strip() # bye espaços em branco dos nomes das colunas

DF["CAUSAFAMILIA"] = DF["CAUSAFAMILIA"].apply(simplificar_familia)
DF["MES"] = DF["MES"].astype(int) 
DF["MES_NOME"] = DF["MES"].map(_BASE_MESES_INICIAIS)
DF["DISTRITO"] = DF["DISTRITO"].str.title()
DF["CONCELHO"] = DF["CONCELHO"].str.title().fillna("Desconhecido")
DF["TIPO"] = DF["TIPO"].fillna("Desconhecido")
DF['DIA'] = pd.to_numeric(DF['DIA'], errors='coerce')


# Listas para filtros e dropdowns
DISTRITOS = ["Todos"] + sorted(DF["DISTRITO"].dropna().unique()) # distritos únicos
ANOS = sorted([int(ano) for ano in DF["ANO"].dropna().unique()]) # anos únicos para o slider

METRICAS_OPCOES = [ 
    {"label": "Número de Incêndios", "value": "NUM_INCENDIOS"},
    {"label": "Área Ardida Total", "value": "AREA_ARDIDA"},
    {"label": "Duração Média", "value": "DURACAO_MEDIA"}
]
METRICAS_OPCOES_MAP = {}
METRICAS_OPCOES_MAP = {opt['value']: opt for opt in METRICAS_OPCOES}

OPCOES_DISTRITOS = [{"label": c, "value": c} for c in DISTRITOS] # para Dropdown

# Opções para os botões de rádio de seleção de mês na sidebar
MES_RADIO_OPCOES = [{"label": html.Span("Todos", id="mes-tooltip-0"), "value": 0}] # Opção "Todos"
for num, nome_curto in MESES_CURTO_RADIO.items():
    MES_RADIO_OPCOES.append({
        "label": html.Span(nome_curto, id=f"mes-tooltip-{num}"),
        "value": num
    })


MAIN_MAP_FIXED_HEIGHT = "500px" # altura fixa para o mapa principal
METEO_MAP_NEW_HEIGHT = "572px" # altura para os mapas meteorológicos (temperatura, humidade, vento)
DEFAULT_CENTER_PT = {"lat": 39.56, "lon": -8.0} # ponto central padrão para os mapas (Portugal Continental)
ZOOM_PORTUGAL = 5.5 # nivel de zoom inicial para Portugal
ZOOM_DISTRITO = 7.5 # zoom ao selecionar um distrito
MAX_ZOOM_CONCELHO = 12 # máximo de zoom para concelhos

TIPOCAUSA_COLOR_MAP = {
    "Intencional": PALETTE["accent_orange"],
    "Negligente": PALETTE["accent_red"],
    "Reacendimento": "#4CAF50",
    "Natural": "#BDBDBD", 
    "Desconhecida": PALETTE["brand_dark"],
    "NÃO ESPECIFICADA": PALETTE["brand_dark"],
}

TITULOS_METRICAS = {
    "NUM_INCENDIOS": "Nº de Incêndios",
    "AREA_ARDIDA": "Área Ardida",
    "DURACAO_MEDIA": "Duração Média dos Incêndios"
}

COLOR_MAP_TIPO = {
    "Florestal": "#2E7D32", 
    "Agrícola": "#795548", 
    "Urbano": "#616161", 
    "Desconhecido": "#BDBDBD" 
}

# para a nuvem de palavras
WORDCLOUD_COLORMAP = cm._colormaps.get_cmap("OrRd") # "Orange-Red"
WORDCLOUD_MIN_COLOR_SCALE = 0.3 # escala mínima de cor para palavras menores
WORDCLOUD_MAX_FONT_SIZE = 60
WORDCLOUD_MIN_FONT_SIZE = 13

# Escalas de cor e intervalos para mapas meteorológicos
TEMP_COLOR_SCALE =[[0.0, "#DAA520"], [0.6, "#FF4500"], [1.0, "#B22222"]] # Amarelo -> Laranja -> Vermelho escuro
TEMP_RANGE = [-5, 45] # Intervalo de temperatura em °C
TEMP_COLOR_SCALE_DENSITY = [[0.0, "#87CEEB"], [0.2, "#DAA520"], [0.4, "#FF4500"], [0.6, "#FF4500"], [1.0, "#B22222"]] 
TEMP_RANGE_DENSITY = [-5, 45]

# Tratamento da coluna VENTOINTENSIDADE e definição do máximo para o slider
MAX_WIND_SPEED = DF['VENTOINTENSIDADE'].max()
MAX_SLIDER_WIND = int(np.ceil(MAX_WIND_SPEED)) if pd.notna(MAX_WIND_SPEED) and MAX_WIND_SPEED > 0 else 60 

# Tamanhos de fonte para controlos de animação de mapas
FONT_SIZE_ANIMATION_VALUE = 13
FONT_SIZE_ANIMATION_BUTTON = 12
FONT_SIZE_ANIMATION_STEP_LABEL = 10

# Tratamento da coluna HUMIDADERELATIVA (clipar valores entre 0 e 100)
DF.loc[DF["HUMIDADERELATIVA"] > 100, "HUMIDADERELATIVA"] = 100
DF.loc[DF["HUMIDADERELATIVA"] < 0, "HUMIDADERELATIVA"] = 0

# Escala de cor e intervalo para o mapa de vento
WIND_COLOR_SCALE = [[0, "#32CD32"], [0.5, "#008000"], [1, "#4F7942"]] # Verde claro -> Verde escuro -> Verde oliva
WIND_RANGE_COLOR = [0, 40]

BASE_SCALE = 0.35
HEAD_WING = 0.15
HEAD_CONN = 0.75
SHAFT_WIDTH = 1.2
HEAD_WIDTH = 0.6
LENGTH_CAP = WIND_RANGE_COLOR[1] # lim pra normalização do comprimento da seta


# Funções e Textos para o Modo de Ajuda

# Função para criar um Div com texto de ajuda formatado
def create_help_text_div(message_html: str, height: Union[str, int], chart_id_for_logging: str) -> html.Div:
    if isinstance(height, str) and "px" in height:
        height_val = height
    elif isinstance(height, int):
        height_val = f"{height}px"
    else:
        height_val = "200px"

    return html.Div(
        html.Div(
            dcc.Markdown(message_html, dangerously_allow_html=False, link_target="_blank"), # Markdown para formatação
            style={ # Estilos para o conteúdo interno (scroll, etc.)
                "width": "100%", 
                "maxHeight": "100%", 
                "overflowY": "auto",
                "paddingRight": "10px", 
                "textAlign": "left"
            }
        ),
        style={ # Estilos para o container principal do texto de ajuda
            "height": height_val, 
            "padding": "15px", 
            "display": "flex",
            "flexDirection": "column", 
            "alignItems": "flex-start", 
            "justifyContent": "flex-start",
            "backgroundColor": PALETTE["card_bg"], 
            "color": PALETTE["font"],
            "fontSize": "13px", 
            "lineHeight": "1.5",
            "border": f"1px dashed {PALETTE.get('brand', '#ffb66d')}", # Borda tracejada
            "borderRadius": "4px", 
            "boxSizing": "border-box",
        },
        className=f"help-text-container help-for-{chart_id_for_logging}" # Classe CSS para possível estilização adicional
    )


HELP_TEXTS = {
    "g-mapa": """
Aqui podes ver onde ocorreram incêndios em Portugal Continental na data selecionada, agrupados por **Distrito** ou **Concelho**.
- Os **círculos** mostram os locais dos incêndios. O **tamanho** de cada círculo indica o valor da métrica que escolheste (Nº de Incêndios, Área Ardida ou Duração Média).
- A **cor** de cada círculo indica se nesse local predominam incêndios florestais ou agrícolas.
- Podes **interagir** com o mapa: clica num círculo para filtrar os outros gráficos para esse local, ou ativa os nomes dos locais para identificá-los mais facilmente.
    """,
    "g-perfil-horario": """
Aqui, podes ver como a métrica selecionada na barra lateral (Nº de Incêndios, Área Ardida Média ou Duração Média) varia ao longo das 24 horas do dia, com base na **hora de início** do incêndio.
- A **linha** indica o valor médio para cada hora de início e a **área sombreada** mostra a variabilidade dos dados em torno da média (desvio padrão).

Este gráfico ajuda a identificar os períodos do dia com maior frequência de início de incêndios ou maior impacto inicial, permitindo entender padrões horários de ignição.
""",
    "pie-cloud-tipo": """
Aqui podes ver como os incêndios se distribuem pelas diferentes **causas**. A informação é apresentada de acordo com a métrica que escolheste (Nº de Incêndios, Área Ardida ou Duração).
- **Cada fatia** do gráfico representa um tipo de causa de incêndio.
- O **tamanho** de cada fatia mostra a importância dessa causa.
- As **percentagens** mostram a proporção exacta de cada tipo de causa.

Este gráfico ajuda-te a perceber rapidamente quais são as principais causas dos incêndios.
    """,
    "pie-cloud-familia": """
Aqui podes ver as **famílias de causa** mais importantes, de acordo com a métrica que escolheste (Nº de Incêndios, Área Ardida ou Duração). As famílias são categorias mais específicas de causas, como "Queimas e Queimadas" ou "Uso de Maquinaria".
- Quanto mais importante é uma família, **maior** é o seu tamanho e mais **forte** é a sua cor.

Esta visualização permite-te identificar facilmente as causas mais significativas dos incêndios.
    """,
    "g-meteo-map": """
Aqui podes ver como uma condição meteorológica (Temperatura, Humidade Relativa ou Vento) estava distribuída geograficamente num **dia específico**. **Nota:** Precisas de selecionar um mês primeiro.
- Para **Temperatura e Humidade**: As cores no mapa mostram os diferentes valores.
- Para o **Vento**: As cores mostram a intensidade e as setas mostram a direção e força.
- Se houver dados diários, um **controlo deslizante** ("Dia") aparecerá abaixo do mapa, permitindo-te ver a evolução ao longo dos dias.
- Podes fazer **zoom** e mover o mapa para ver melhor.

Este mapa ajuda-te a relacionar as condições meteorológicas com a ocorrência de incêndios em cada região.
    """,
    "g-violin": """
Aqui podes ver quais eram os valores de Temperatura, Humidade Relativa e Vento quando os incêndios ocorreram.
- Cada **"violino"** representa uma condição meteorológica. A sua forma mostra como os dados estão distribuídos:
    - A **largura** do violino em qualquer ponto indica a frequência de incêndios com esse valor específico. Partes mais largas significam que mais incêndios ocorreram nessas condições. Ao passar o rato sobre o gráfico, por vezes poderás ver um valor técnico chamado 'kde'; este valor está diretamente relacionado com a largura do violino: um 'kde' maior significa que o violino é mais largo nesse ponto, indicando uma maior concentração de incêndios.
- A **caixa** dentro de cada violino (chamada *boxplot*) resume os dados principais:
    - A linha no meio da caixa: o valor central (mediana) – metade dos incêndios ocorreram com valores abaixo desta linha e metade acima.
    - Os limites superior e inferior da caixa: representam o intervalo onde se encontram os 50% centrais dos casos (entre o 1º e o 3º quartil).
    - As linhas que se estendem da caixa (chamadas "bigodes" ou *whiskers*): abrangem a maioria dos restantes casos (tipicamente, excluem apenas os valores muito extremos ou *outliers*).
- A linha horizontal mais pequena (por vezes tracejada) dentro do violino ou da caixa representa o valor médio.

Este gráfico ajuda-te a perceber em que condições meteorológicas os incêndios costumam ocorrer e qual a dispersão desses valores.
    """,
    "g-relacao-metricas": """
Aqui podes ver como as várias medidas de incêndios se relacionam ao longo do ano.
- O **eixo horizontal** mostra os meses do ano (Janeiro a Dezembro).
- As **barras** mostram o **Número de Incêndios** em cada mês.
- A **cor** das barras indica a **Área Ardida Total** nesse mês. Cores mais escuras significam maior área ardida.
- A **linha** mostra a **Duração Média dos Incêndios** em horas para cada mês.

Este gráfico ajuda-te a ver quando os incêndios são mais frequentes, intensos e duradouros ao longo do ano.
    """,
    "g-scatter-meteo": """
Este gráfico mostra a relação entre **Temperatura** e **Humidade Relativa** nos locais onde ocorreram incêndios.
- Cada **ponto** representa um incêndio.
- O **eixo horizontal** (X) indica a Temperatura (°C).
- O **eixo vertical** (Y) indica a Humidade Relativa (%).
- A **cor** do ponto indica o tipo de incêndio (Florestal ou Agrícola).
- O **tamanho** do ponto indica a intensidade do vento (km/h) no momento do incêndio. Pontos maiores significam vento mais forte.
- Podes usar o **filtro de Vento** acima do gráfico para mostrar apenas incêndios que ocorreram dentro de um intervalo específico de intensidade de vento.

Este gráfico ajuda a identificar combinações de condições meteorológicas (temperatura, humidade, vento) que podem ser mais propícias a incêndios de diferentes tipos.
    """
}

# Função para criar uma figura vazia com uma mensagem bonitinha ----< usada quando não há dados
def create_empty_figure(message: str = "sem dados para exibir.", height: Optional[int] = None):
    fig = go.Figure()
    fig.add_annotation(
        text=f"<span style='font-size:13px; color:{PALETTE.get('font', '#2C3E50')};'>{message.replace(chr(10), '<br>')}</span>", # Permite quebras de linha na mensagem
        x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, # pos central
        align="center", xanchor="center", yanchor="middle"
    )
    default_height = int(METEO_MAP_NEW_HEIGHT.replace("px", ""))
    fig_height = height if height is not None else default_height

    fig.update_layout(
        paper_bgcolor=PALETTE.get("card_bg", "#FFFFFF"),
        plot_bgcolor=PALETTE.get("card_bg", "#FFFFFF"),
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=fig_height
    )
    return fig


# Formata duração em minutos---  "2 dias, 3 horas e 15 minutos"
def format_duration_dhm_verbose_refined(total_minutes):
    if total_minutes is None or pd.isna(total_minutes) or total_minutes < 0: return "n/a"
    try:
        total_minutes = int(round(total_minutes))
        if total_minutes == 0: 
            return "0 minutos"
        
        days = total_minutes // (24 * 60)
        remaining_minutes_after_days = total_minutes % (24 * 60)
        hours = remaining_minutes_after_days // 60
        minutes = remaining_minutes_after_days % 60

        parts = []
        
        if days > 0: 
            parts.append(f"{days} dia{'s' if days > 1 else ''}")
        if hours > 0: 
            parts.append(f"{hours} hora{'s' if hours > 1 else ''}")
        if minutes > 0: 
            parts.append(f"{minutes} minuto{'s' if minutes > 1 else ''}")
        if not parts: 
            return "0 minutos" 
        
        # Concatena as partes de forma gramaticalmente correta
        if len(parts) == 1: 
            return parts[0]
        elif len(parts) == 2: 
            return f"{parts[0]} e {parts[1]}"
        elif len(parts) == 3: 
            return f"{parts[0]}, {parts[1]} e {parts[2]}"
        return ", ".join(parts) 
    except (ValueError, TypeError): return "n/a" 
    

# Formata duração em minutos para uma string compacta (ex: "50h30m")
def format_duration_hm(total_minutes):
    if total_minutes is None or pd.isna(total_minutes) or total_minutes < 0: 
        return "n/a"
    try:
        total_minutes = int(round(total_minutes))
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h{minutes:02d}m" # Formato XhYYm (minutos com 2 dígitos)
    except (ValueError, TypeError): return "n/a"


# Gráfico de Relação entre Métricas (Nº Incêndios, Área Ardida, Duração Média por Mês)
def fig_relacao_metricas(df_data: pd.DataFrame, ano_selecionado: int, active_filter_name: str, altura_grafico: int = 250):
    if df_data.empty:
        return create_empty_figure(f"Sem dados para {active_filter_name.lower()} em {ano_selecionado}<br>para o gráfico de relação.", height=altura_grafico)

    # Agrega os dados por mês
    df_agg = df_data.groupby("MES").agg(
        NUM_INCENDIOS=("id", "count"), 
        AREA_ARDIDA_TOTAL=("AREATOTAL", "sum"), 
        DURACAO_SOMA_MIN=("DURACAO", "sum"), 
        DURACAO_CONTAGEM_VALIDA=("DURACAO", lambda x: x.notna().sum())
    ).reset_index()

    # Calcula Duração Média em minutos e horas
    df_agg["DURACAO_MEDIA_MIN"] = df_agg.apply(
        lambda r: (r["DURACAO_SOMA_MIN"]/r["DURACAO_CONTAGEM_VALIDA"])
        if r["DURACAO_CONTAGEM_VALIDA"] > 0 else 0, axis=1
    )
    df_agg["DURACAO_MEDIA_HORAS"] = df_agg["DURACAO_MEDIA_MIN"] / 60.0
    df_agg["MES_NOME"] = df_agg["MES"].map(MESES_CURTO_RADIO) # add nome curto do mês
    df_agg = df_agg.sort_values(by="MES") # Ordena por mês
    df_agg["DURACAO_MEDIA_HM_STR"] = df_agg["DURACAO_MEDIA_MIN"].apply(format_duration_hm) # duração para hover
    df_agg["MES_EXTENSO"] = df_agg["MES"].map(_BASE_MESES_EXTENSO) # nome extenso do mês para hover
    
    customdata_bar = np.stack((df_agg['MES_EXTENSO'], df_agg['AREA_ARDIDA_TOTAL'], df_agg['DURACAO_MEDIA_HM_STR']), axis=-1)

    if df_agg.empty or df_agg["NUM_INCENDIOS"].sum() == 0:
        return create_empty_figure(f"Sem dados de incêndios para {active_filter_name.lower()} em {ano_selecionado}<br>após agregação.", height=altura_grafico)

    fig = make_subplots(specs=[[{"secondary_y": True}]]) 
    
    min_area = 0
    max_area = df_agg["AREA_ARDIDA_TOTAL"].max() if not df_agg["AREA_ARDIDA_TOTAL"].empty else 1
    if max_area == 0: 
        max_area = 1 
    FONT_SIZE_LEGEND = 11 

    # Trace de Barras: Nº de Incêndios (cor pela Área Ardida)
    bar_trace = go.Bar(
        x=df_agg["MES_NOME"],
        y=df_agg["NUM_INCENDIOS"],
        name="Nº de Incêndios",
        marker=dict(
            color=df_agg["AREA_ARDIDA_TOTAL"], # Cor da barra baseada na área ardida
            colorscale='OrRd', # Escala de cores Laranja-Vermelho
            cmin=min_area, cmax=max_area,
            colorbar=dict( 
                title=dict(text="Área Ardida (ha)", font=dict(size=FONT_SIZE_LEGEND_TITLE, color=PALETTE["font"])),
                orientation="h", x=0.5, xanchor="center", y=-0.3, yanchor="top",
                thickness=12, len=0.75, tickfont=dict(size=FONT_SIZE_TICK_LABEL, color=PALETTE["font"]), outlinewidth=0
            ),
            showscale=True # Mostra a colorbar
        ),
        customdata=customdata_bar, # Dados para o hovertemplate
        hovertemplate="<b>%{customdata[0]}</b><br>Nº Incêndios: %{y:.0f}<br>Área Ardida: %{customdata[1]:,.0f} ha<br>Duração Média: %{customdata[2]}<extra></extra>",
        opacity=0.9, hoverlabel=dict(bgcolor=PALETTE["card_bg"])
    )
    fig.add_trace(bar_trace, secondary_y=False) # Adiciona ao eixo Y primário

    # Trace de Linha: Duração Média (horas)
    line_trace = go.Scatter(
        x=df_agg["MES_NOME"],
        y=df_agg["DURACAO_MEDIA_HORAS"],
        name="Duração Média (h)",
        mode='lines+markers', # Linhas com marcadores
        line=dict(color=PALETTE["brand_dark"], width=2.5),
        marker=dict(color=PALETTE["brand_dark"], size=8, symbol='circle'),
        customdata=np.stack((df_agg['NUM_INCENDIOS'], df_agg['AREA_ARDIDA_TOTAL'], df_agg['DURACAO_MEDIA_HM_STR']), axis=-1),
        hoverinfo="skip" # O hover deste trace é combinado com o das barras através do hovermode="closest" e do hovertemplate das barras
    )
    fig.add_trace(line_trace, secondary_y=True) # Adiciona ao eixo Y secundário
    fig.update_traces(hoverlabel=dict(namelength=-1)) # Mostra nome completo no hover da legenda

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Mês", font=dict(size=FONT_SIZE_AXIS_TITLE, color=PALETTE["font"])),
            tickfont=dict(size=FONT_SIZE_TICK_LABEL, color=PALETTE["font"])),
        plot_bgcolor=PALETTE["card_bg"], paper_bgcolor=PALETTE["card_bg"], font_color=PALETTE["font"],
        legend=dict( # Configuração da legenda principal
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=FONT_SIZE_LEGEND), bgcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=20, b=50), hovermode="closest", height=altura_grafico # Hovermode "closest" combina hovers
    )
    fig.update_yaxes(
        title=dict(text="Nº de Incêndios", font=dict(size=FONT_SIZE_AXIS_TITLE, color=PALETTE["font"]), standoff=10),
        secondary_y=False, tickfont=dict(size=FONT_SIZE_TICK_LABEL, color=PALETTE["font"]), gridcolor='rgba(200,200,200,0.3)'
    )
    fig.update_yaxes(
        title=dict(text="Duração Média (horas)", font=dict(size=FONT_SIZE_AXIS_TITLE, color=PALETTE["font"]), standoff=10),
        secondary_y=True, tickfont=dict(size=FONT_SIZE_TICK_LABEL, color=PALETTE["font"]), showgrid=False,
        range=[0, df_agg["DURACAO_MEDIA_HORAS"].max() * 1.1 if not df_agg.empty and df_agg["DURACAO_MEDIA_HORAS"].max() > 0 else 5] # Define range do eixo Y secundário
    )
    return fig

# Função de cor para a nuvem de palavras (baseada no tamanho da fonte)
def wordcloud_color_func(word, font_size, position, orientation, font_path, random_state):
    if font_size is None or WORDCLOUD_MAX_FONT_SIZE is None or WORDCLOUD_MIN_FONT_SIZE is None:
        scale = np.random.uniform(WORDCLOUD_MIN_COLOR_SCALE, 1.0) # Escala aleatória se tamanhos não definidos
    else:
        if WORDCLOUD_MAX_FONT_SIZE == WORDCLOUD_MIN_FONT_SIZE: 
            font_ratio = 1.0
        else: 
            font_ratio = (font_size - WORDCLOUD_MIN_FONT_SIZE) / (WORDCLOUD_MAX_FONT_SIZE - WORDCLOUD_MIN_FONT_SIZE)
        scale = WORDCLOUD_MIN_COLOR_SCALE + font_ratio * (1.0 - WORDCLOUD_MIN_COLOR_SCALE)
        scale = np.clip(scale, WORDCLOUD_MIN_COLOR_SCALE, 1.0) 
    rgba = WORDCLOUD_COLORMAP(scale)
    r, g, b = [int(255 * x) for x in rgba[:3]] 
    return (r, g, b) # Retorna tupla RGB


def fig_radar_causas(df_chart_data, active_filter_name_for_title, ano, mes_val):
    if df_chart_data.empty or "TIPOCAUSA" not in df_chart_data.columns:
        return create_empty_figure(f"Sem dados de tipos de causa para {active_filter_name_for_title.lower()}<br>({get_time_period_string(ano, mes_val).lower()}).")
    contagem_causas = df_chart_data["TIPOCAUSA"].value_counts().reset_index(name="Contagem") # Conta ocorrências por tipo de causa
    if contagem_causas.empty: 
        return create_empty_figure(f"Sem dados de tipos de causa para {active_filter_name_for_title.lower()}<br>({get_time_period_string(ano, mes_val).lower()}).")
    
    fig = go.Figure(go.Scatterpolar(
        r=contagem_causas["Contagem"], # Valores (raio)
        theta=contagem_causas["TIPOCAUSA"], # Categorias (ângulo)
        fill="toself", name="causas", # Preenche a área
        line=dict(color=PALETTE["accent_red"],width=3), fillcolor="rgba(231,76,60,0.25)"
    ))
    fig.update_layout(
        title=None, 
        polar=dict(radialaxis=dict(visible=True, range=[0, contagem_causas["Contagem"].max()], gridcolor="lightgray", tickfont=dict(size=FONT_SIZE_TICK_LABEL)),
                   angularaxis=dict(tickfont=dict(size=FONT_SIZE_TICK_LABEL), gridcolor="lightgray")),
        font=dict(color=PALETTE["font"]), showlegend=False, 
        paper_bgcolor=PALETTE["card_bg"], plot_bgcolor=PALETTE["card_bg"], 
        margin=dict(l=45,r=45,t=15,b=15)
    )
    return fig

# Helper para obter uma string formatada do período (ano e mês)
def get_time_period_string(ano: int, mes_val: int) -> str:
    if mes_val == 0: 
        return f"ano de {ano}" # Se "Todos os Meses"
    return f"{MESES_EXTENSO.get(mes_val, '')} de {ano}" # Para um mês específico

# Helper para obter o tipo de incêndio mais frequente (usado na agregação do mapa principal)
def get_most_frequent_tipo(series: pd.Series) -> str:
    if series.empty: 
        return "desconhecido"
    modes = series.mode() # Retorna os valores mais frequentes (pode haver mais de um)
    return modes[0] if not modes.empty else "desconhecido" # Retorna o primeiro, se houver



# Função para criar o conteúdo principal do dashboard (gráficos e controlos)
def create_main_content():
    meteo_map_h_str = METEO_MAP_NEW_HEIGHT
    chart_h_str = "215px" # Altura para gráficos menores como o pie/cloud
    perfil_horario_height_str = "200px"
    violin_height_str = "305px"
    scatter_meteo_graph_height_str = "305px"
    relacao_metricas_height_str = "250px"
    main_map_fixed_height_str = MAIN_MAP_FIXED_HEIGHT

    # Card: Perfil Horário
    card_perfil_horario = dbc.Card(dbc.CardBody([
        html.H5("Evolução Horária do Início dos Incêndios", style={ # Título do card
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "fontWeight": "bold",
            "color": PALETTE["font"], "textAlign": "center",
            "marginBottom": "2px", "marginTop": "2px"
        }),
        html.Div(id="display-area-perfil-horario") # Container para o gráfico do perfil horário
    ]))

    # Card: Gráfico de Pizza / Nuvem de Palavras (Causas)
    card_pie_cloud = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H5(id="pie-cloud-title", style={ # Título dinâmico
                "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "fontWeight": "bold",
                "color": PALETTE["font"], "textAlign": "center", "marginBottom": "4px"
            }), width=12)
        ], className="mb-2"),
        dbc.Row([ # Controlos para alternar entre Pizza (Tipo de Causa) e Nuvem (Família de Causa)
            dbc.Col(html.Div([
                dbc.ButtonGroup([ # Botões segmentados
                    dbc.Button("Por Tipo", id="btn-pie-tipo", color="secondary", outline=True, active=True, className="segmented-btn"),
                    dbc.Button("Por Família", id="btn-pie-familia", color="secondary", outline=True, className="segmented-btn"),
                ], className="segmented-control-container d-flex", id="segmented-pie-control"),
                dcc.Store(id="radio-pie-cloud-selector", data="tipo") # Armazena a seleção atual (tipo/familia)
            ], style={"textAlign": "center", "paddingTop": "5px"}))
        ], className="mb-2", justify="center"),
        html.Div(id="pie-cloud-display-area", style={"height": chart_h_str }) # Container para o gráfico
    ]))

    # Card: Mapa Meteorológico (Temperatura, Humidade, Vento)
    card_meteo_map = dbc.Card(dbc.CardBody([
        html.H5(id="meteo-map-title-dynamic", style={ # Título dinâmico
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "fontWeight": "bold",
            "color": PALETTE["font"], "textAlign": "center",
            "marginBottom": "2px", "marginTop": "7px"
        }),
        dbc.Row([ # Controlos para selecionar a variável meteorológica
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button("Temperatura", id="btn-meteo-temp", color="secondary", outline=True, active=True, className="segmented-btn"),
                    dbc.Button("Humidade", id="btn-meteo-hum", color="secondary", outline=True, className="segmented-btn"),
                    dbc.Button("Vento", id="btn-meteo-vento", color="secondary", outline=True, className="segmented-btn"),
                ], className="segmented-control-container d-flex")
            ], style={"marginBottom": "6px", "textAlign": "center", "marginTop": "15px"})
        ], justify="center", className="mb-1"),
        dcc.Store(id="rd-meteo-var", data="TEMPERATURA"), # Armazena a variável meteo selecionada
        html.Div(id="display-area-meteo-map") # Container para o mapa meteo
    ]))

    # Card: Gráfico de Violino (Distribuição das Variáveis Meteorológicas)
    card_violin = dbc.Card(dbc.CardBody([
        html.H5("Distribuição das Variáveis Meteorológicas", style={
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "fontWeight": "bold",
            "color": PALETTE["font"], "textAlign": "center",
            "marginBottom": "2px", "marginTop": "2px"
        }),
        html.Div(id="display-area-violin") # Container para o gráfico de violino
    ]))

    # Card: Gráfico de Dispersão (Temperatura vs Humidade, tamanho por Vento)
    card_scatter_meteo = dbc.Card(dbc.CardBody([
        html.H5("Temperatura vs Humidade Relativa por Intensidade de Vento", style={
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "fontWeight": "bold",
            "color": PALETTE["font"], "textAlign": "center", "marginBottom": "15px"
        }),
        html.Div([ # RangeSlider para filtrar por intensidade do vento
            html.Label("Vento (km/h):", style={
                'fontWeight': 'normal', 'fontSize': f'{FONT_SIZE_AXIS_TITLE}px',
                'color': PALETTE["font"], 'display': 'block',
                'textAlign': 'center', 'marginBottom': '8px'
            }),
            dcc.RangeSlider(
                id='rangeslider-scatter-wind-filter', min=0, max=MAX_SLIDER_WIND, step=1,
                marks={i: {'label': str(i), 'style': {'fontSize': '9px'}} for i in range(0, MAX_SLIDER_WIND + 1, 10)}, # Marcas a cada 10 km/h
                value=[0, MAX_SLIDER_WIND], # Valor inicial (todo o intervalo)
                tooltip={"placement": "bottom", "always_visible": False},
                className="slider-vento-custom", updatemode='mouseup' # Atualiza ao largar o rato
            )
        ], style={"marginBottom": "5px", "paddingLeft": "10px", "paddingRight": "10px"}),
        html.Div(id="display-area-scatter-meteo") # Container para o gráfico de dispersão
    ]), style={"height": "auto", "paddingTop": "15px", "paddingBottom": "15px"})

    # Card: Mapa Principal de Incêndios
    main_map_card = dbc.Card(dbc.CardBody([
        html.H5(id="mapa-dynamic-title", style={ # Título dinâmico
            "color": PALETTE["font"], "fontWeight": "bold",
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "marginBottom": "2px", "textAlign": "center"
        }),
        html.P("Clica num círculo/localidade no mapa para ver detalhes e filtrar os restantes gráficos", style={ # Instrução para o utilizador
            "fontSize": "0.75rem", "color": "#6c757d", "textAlign": "center", "marginBottom": "10px"
        }),
        dbc.Row([ # Controlos do mapa: granularidade e toggle de nomes
            dbc.Col(html.Div([
                dcc.Dropdown( # Dropdown para selecionar granularidade (Distrito/Concelho)
                    id="dd-map-granularity",
                    options=[{"label": "Distrito", "value": "DISTRITO"}, {"label": "Concelho", "value": "CONCELHO"}],
                    value="DISTRITO", placeholder="nível mapa", clearable=False, searchable=False,
                    style={"width": "100px", "fontSize": "0.75rem", "padding": "0px 8px",
                           "marginRight": "20px", "marginTop": "-5px", "marginBottom": "-5px", "height": "25px"}
                ),
                dbc.Switch(id="map-text-toggle", value=False, persistence=True, persistence_type='session', # Toggle para mostrar/esconder nomes no mapa
                           className="me-1", style={"display": "inline-block", "verticalAlign": "middle", "accentColor": "#999999"}),
                html.Span(id="map-text-toggle-label", children="nomes distritos", # Label do toggle (dinâmico)
                          style={"fontSize": "0.75rem", "color": PALETTE["font"], "verticalAlign": "middle"})
            ], className="d-flex align-items-center"), width="auto")
        ], justify="start", align="center", className="mb-2 g-2", style={"minHeight": "45px", "padding": "0 5px"}),
        html.Div(id="display-area-mapa") # Container para o mapa principal
    ]), style={"backgroundColor": PALETTE["card_bg"]})

    # Card: Gráfico de Relação entre Métricas (barras e linha)
    card_relacao_metricas = dbc.Card(dbc.CardBody([
        html.H5("Relação entre Incêndios, Área Ardida e Duração", style={
            "fontSize": f"{FONT_SIZE_CHART_TITLE}px", "textAlign": "center",
            "color": PALETTE["font"], "fontWeight": "bold", "marginBottom": "0px"
        }),
        html.Div(id="display-area-relacao-metricas") # Container para o gráfico
    ]), className="mb-0")

    # Armazena as alturas dos gráficos em dcc.Store para serem acessíveis no modo de ajuda
    store_heights = {
        'store-perfil-horario-height': int(perfil_horario_height_str.replace("px","")),
        'store-violin-height': int(violin_height_str.replace("px","")),
        'store-scatter-meteo-height': int(scatter_meteo_graph_height_str.replace("px","")),
        'store-relacao-metricas-height': int(relacao_metricas_height_str.replace("px","")),
        'store-main-map-height': int(main_map_fixed_height_str.replace("px","")),
        'store-meteo-map-height': int(meteo_map_h_str.replace("px","")),
        'store-pie-cloud-height': int(chart_h_str.replace("px",""))
    }

    # Estrutura do layout principal com Rows e Cols do Bootstrap
    return html.Div([
        html.Div([ # Header com título e botão de reset
            dbc.Button(
                [html.I(className="fas fa-undo me-1"), "Reset"], id="btn-reset-filtros", color="danger",
                outline=True, className="btn-sm",
                style={"position": "absolute", "top": "12px", "right": "10px", "zIndex": "10", "fontSize": "0.8rem"}
            ),
            html.H1("Incêndios em Portugal Continental", style={"color": PALETTE["brand_dark"], "marginTop": "12px", "marginBottom": "5px", "fontWeight": "bold", "textAlign": "center", "fontSize": "2rem"}),
        ], style={"position": "relative", "marginBottom": "0px"}),
        html.H4(id="subtitulo-dinamico", style={"color": PALETTE["font"], "marginBottom": "20px", "fontWeight": "normal", "textAlign": "center", "fontSize": "1.1rem"}), # Subtítulo dinâmico
        dbc.Row([ # Layout em duas colunas principais
            # Coluna da Esquerda (Mapa Principal e Scatter Meteo)
            dbc.Col([main_map_card, html.Div(className="mt-2"), card_scatter_meteo], md=4, style={"paddingRight": "5px"}),
            # Coluna da Direita (Restantes Gráficos)
            dbc.Col([
                dbc.Row([dbc.Col(card_perfil_horario, md=12)], className="mb-2"),
                dbc.Row([
                    dbc.Col([card_pie_cloud, html.Div(className="mt-2"), card_violin], md=6),
                    dbc.Col(card_meteo_map, md=6)
                ], className="g-2 mb-2"),
                dbc.Row([dbc.Col(card_relacao_metricas, md=12)], className="g-2 mb-2")
            ], md=8, style={"paddingLeft": "5px"})
        ], className="g-2"), # g-2 para gutters (espaçamento)
        # Adiciona os dcc.Store para as alturas
        *[dcc.Store(id=store_id, data=height_val) for store_id, height_val in store_heights.items()]
    ], style={"marginLeft": "240px", "backgroundColor": PALETTE["bg"], "minHeight": "100vh", "paddingRight": "8px", "paddingLeft": "8px"}) # Margem para a sidebar


# Mapa Principal de Incêndios (por Distrito ou Concelho)
def fig_mapa(df_year_month_filtered: pd.DataFrame, metric: str, ano: int, mes_val: int, show_text_labels: bool = False, granularity: str = "DISTRITO"):
    if granularity not in ["DISTRITO", "CONCELHO"] or granularity is None:
        granularity = "DISTRITO" # Default para Distrito

    time_period_str = get_time_period_string(ano, mes_val) # String do período para mensagens
    map_height = int(MAIN_MAP_FIXED_HEIGHT.replace("px","")) # Altura do mapa

    if df_year_month_filtered.empty:
         return create_empty_figure(f"Sem dados para exibir no mapa<br>({granularity.lower()}) – {time_period_str}", height=map_height)

    # Remove dados sem LAT, LON ou a granularidade selecionada (Distrito/Concelho)
    df_mapa_base = df_year_month_filtered.dropna(subset=["LAT", "LON", granularity])

    if df_mapa_base.empty:
        return create_empty_figure(f"Sem dados de localização para exibir no mapa<br>({granularity.lower()}) – {time_period_str}", height=map_height)

    # Agregação base: média de LAT/LON, Duração Média, Tipo Predominante por granularidade
    agg_base = df_mapa_base.groupby(granularity).agg(
        LAT=("LAT", "mean"), LON=("LON", "mean"),
        DURACAO_MEDIA_RAW_MIN=("DURACAO", "mean"), TIPO_PREDOMINANTE=("TIPO", get_most_frequent_tipo)
    ).reset_index()

    # Calcula contagens de incêndios por tipo (Florestal, Agrícola)
    counts_por_tipo = df_mapa_base.groupby([granularity, "TIPO"]).size().unstack(fill_value=0)
    for tipo in ["Florestal", "Agrícola", "Urbano", "Desconhecido"]: # Garante que todas as colunas de tipo existem
        if tipo not in counts_por_tipo.columns: 
            counts_por_tipo[tipo] = 0
    counts_por_tipo = counts_por_tipo.rename(columns={ # Renomeia colunas para clareza
        "Florestal": "NUM_INCENDIOS_FLORESTAL", "Agrícola": "NUM_INCENDIOS_AGRICOLA",
        "Urbano": "NUM_INCENDIOS_URBANO", "Desconhecido": "NUM_INCENDIOS_DESCONHECIDO"
    }).reset_index()
    num_incendios_cols_for_total = ["NUM_INCENDIOS_FLORESTAL", "NUM_INCENDIOS_AGRICOLA", "NUM_INCENDIOS_URBANO", "NUM_INCENDIOS_DESCONHECIDO"]
    existing_num_cols = [col for col in num_incendios_cols_for_total if col in counts_por_tipo.columns]
    counts_por_tipo["NUM_INCENDIOS_TOTAL"] = counts_por_tipo[existing_num_cols].sum(axis=1) if existing_num_cols else 0 # Soma para total de incêndios

    # Calcula área ardida por tipo
    area_por_tipo = df_mapa_base.groupby([granularity, "TIPO"])["AREATOTAL"].sum().unstack(fill_value=0)
    for tipo in ["Florestal", "Agrícola", "Urbano", "Desconhecido"]: # Garante colunas
        if tipo not in area_por_tipo.columns: 
            area_por_tipo[tipo] = 0
    area_por_tipo = area_por_tipo.rename(columns={ # Renomeia colunas
        "Florestal": "AREA_ARDIDA_FLORESTAL", "Agrícola": "AREA_ARDIDA_AGRICOLA",
    }).reset_index()
    area_cols_for_total = [col for col in area_por_tipo.columns if "AREA_ARDIDA_" in col or col in ["Urbano", "Desconhecido", "AREA_ARDIDA_URBANO", "AREA_ARDIDA_DESCONHECIDO"]]
    existing_area_cols = [col for col in area_cols_for_total if col in area_por_tipo.columns]
    area_por_tipo["AREA_ARDIDA_TOTAL"] = area_por_tipo[existing_area_cols].sum(axis=1) if existing_area_cols else 0 # Soma para área total

    # Junta todas as agregações
    agg_level_data = agg_base.merge(counts_por_tipo, on=granularity, how="left").merge(area_por_tipo, on=granularity, how="left")
    # Preenche NaNs com 0 para colunas numéricas e formata duração
    cols_to_fill = [c for c in agg_level_data.columns if "NUM_INCENDIOS_" in c or "AREA_ARDIDA_" in c or "DURACAO_MEDIA_RAW_MIN" in c]
    for col in cols_to_fill:
        if col in agg_level_data: 
            agg_level_data[col] = pd.to_numeric(agg_level_data[col], errors='coerce').fillna(0)
        else: agg_level_data[col] = 0 # Adiciona coluna com 0 se não existir
    agg_level_data["DURACAO_MEDIA_HM_STR"] = agg_level_data["DURACAO_MEDIA_RAW_MIN"].apply(format_duration_dhm_verbose_refined)

    # Define a métrica para o tamanho dos círculos no mapa
    size_metric_col = {"NUM_INCENDIOS": "NUM_INCENDIOS_TOTAL", "AREA_ARDIDA": "AREA_ARDIDA_TOTAL", "DURACAO_MEDIA": "DURACAO_MEDIA_RAW_MIN"}.get(metric, "NUM_INCENDIOS_TOTAL")
    text_labels_on_map = agg_level_data[granularity] if show_text_labels else None # Define se mostra texto no mapa

    # Colunas para customdata (informação no hover)
    custom_data_cols = [granularity, "DURACAO_MEDIA_HM_STR", "NUM_INCENDIOS_TOTAL", "AREA_ARDIDA_TOTAL", "TIPO_PREDOMINANTE", "NUM_INCENDIOS_FLORESTAL", "NUM_INCENDIOS_AGRICOLA"]
    for col_name in custom_data_cols: # Garante que todas as colunas de customdata existem
        if col_name not in agg_level_data.columns:
            if "NUM_INCENDIOS_" in col_name or "AREA_ARDIDA_" in col_name: agg_level_data[col_name] = 0
            elif col_name == "DURACAO_MEDIA_HM_STR": agg_level_data[col_name] = "n/a"
            elif col_name == "TIPO_PREDOMINANTE": agg_level_data[col_name] = "Desconhecido"

    # Cria o mapa de dispersão 
    fig = px.scatter_mapbox(
        agg_level_data, lat="LAT", lon="LON",
        size=agg_level_data[size_metric_col] if size_metric_col in agg_level_data and pd.api.types.is_numeric_dtype(agg_level_data[size_metric_col]) and agg_level_data[size_metric_col].sum() > 0 else None, # Tamanho do círculo
        size_max=18, color="TIPO_PREDOMINANTE", color_discrete_map=COLOR_MAP_TIPO, # Cor pelo tipo predominante
        hover_name=granularity, custom_data=agg_level_data[custom_data_cols], text=text_labels_on_map
    )
    
    # Define o template do hover dinamicamente com base na métrica
    additional_counts_info = ("<br>Nº Florestal: %{customdata[5]:.0f}<br>Nº Agrícola: %{customdata[6]:.0f}") # Informação adicional de contagens
    if metric == "NUM_INCENDIOS": 
        hovertemplate_parts = ["<b>%{customdata[0]}</b><br>Nº Incêndios Total: %{customdata[2]:.0f}", additional_counts_info, "<extra></extra>"]
    elif metric == "AREA_ARDIDA": 
        hovertemplate_parts = ["<b>%{customdata[0]}</b><br>Área Ardida Total: %{customdata[3]:,.0f} ha", additional_counts_info, "<extra></extra>"]
    elif metric == "DURACAO_MEDIA": 
        hovertemplate_parts = ["<b>%{customdata[0]}</b><br>Duração Média: %{customdata[1]}", additional_counts_info, "<extra></extra>"]
    else: hovertemplate_parts = ["<b>%{customdata[0]}</b><br>Tipo Predominante: %{customdata[4]}", additional_counts_info, "<extra></extra>"] # Fallback

    fig.update_traces(
        hovertemplate="".join(hovertemplate_parts), # Define o hover
        textposition='top center' if show_text_labels else None, # Posição do texto, se mostrado
        textfont=dict(size=FONT_SIZE_MAP_TEXT, color=PALETTE["font"]) if show_text_labels else None,
        selected=dict(marker=dict(opacity=1, size=22)), unselected=dict(marker=dict(opacity=0.6)), # Estilos para seleção
        marker=dict(sizemin=3) # Tamanho mínimo dos círculos
    )
    fig.update_layout(
        title=None, autosize=True, height=map_height, margin=dict(l=0, r=0, t=0, b=0), # Layout geral
        mapbox=dict(style="carto-positron", center={"lat": 39.5, "lon": -8.0}, zoom=5.56), # Estilo e centro do mapa base
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["card_bg"], font_color=PALETTE["font"],
        legend=dict( # Configuração da legenda
            title=dict(text="Tipo Predom.", font=dict(size=FONT_SIZE_LEGEND_TITLE)), orientation="h",
            yanchor="bottom", y=0.01, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.75)", # Posição e estilo
            borderwidth=0, font=dict(size=FONT_SIZE_LEGEND_ITEM), itemsizing='constant'
        ), uirevision='map_layout_v3' # uirevision para manter zoom/pan entre atualizações parciais
    )
    return fig

# Gráfico de Perfil Horário (variação da métrica ao longo do dia)
def fig_perfil_horario(df_chart_data: pd.DataFrame, metric: str, active_filter_name_for_title: str, ano: int, mes_val: int, height: int = 200) -> go.Figure:
    time_period = get_time_period_string(ano, mes_val) # String do período
    empty_msg_base = f"Sem dados para perfil horário<br>({active_filter_name_for_title.lower()} - {time_period.lower()})."
    if df_chart_data.empty: 
        return create_empty_figure(empty_msg_base, height=height)
    
    df_profile_data = df_chart_data.dropna(subset=["HORA"]) # Remove dados sem HORA
    if df_profile_data.empty: 
        return create_empty_figure(empty_msg_base, height=height)
    
    df_profile_data = df_profile_data.copy() # Evita SettingWithCopyWarning
    df_profile_data["HORA"] = pd.to_numeric(df_profile_data["HORA"], errors='coerce') # Converte HORA para numérico
    df_profile_data = df_profile_data.dropna(subset=["HORA"])
    if df_profile_data.empty: 
        return create_empty_figure(f"Dados de hora inválidos ou ausentes<br>após limpeza para perfil horário<br>({active_filter_name_for_title.lower()} - {time_period.lower()}).", height=height)

    df_profile_data["HORA"] = df_profile_data["HORA"].astype(int).clip(0, 23)
    percentage_band_factor = 0.10
    
    agg_config = {"VALOR_MEAN": ("id", "count") if metric == "NUM_INCENDIOS" else ("AREATOTAL", "mean") if metric == "AREA_ARDIDA" else ("DURACAO", "mean"),
                  "VALOR_STD": ("AREATOTAL", "std") if metric == "AREA_ARDIDA" else ("DURACAO", "std") if metric == "DURACAO_MEDIA" else None, # Desvio padrão
                  "COUNT_FOR_STD": ("AREATOTAL", "count") if metric == "AREA_ARDIDA" else ("DURACAO", "count") if metric == "DURACAO_MEDIA" else None} # Contagem para std
    agg_config = {k: v for k, v in agg_config.items() if v is not None} # Remove chaves com valor None
    agg_per_hour = df_profile_data.groupby("HORA").agg(**agg_config).reset_index()

    if agg_per_hour.empty or "VALOR_MEAN" not in agg_per_hour.columns or agg_per_hour["VALOR_MEAN"].sum() == 0:
        return create_empty_figure(f"Sem dados de hora válidos para agregar<br>({active_filter_name_for_title.lower()} - {time_period.lower()}).", height=height)

    # Calcula ou ajusta o desvio padrão
    if metric == "NUM_INCENDIOS": 
        agg_per_hour["VALOR_STD"] = agg_per_hour.get("VALOR_MEAN", 0) * percentage_band_factor # Banda de 10% para Nº Incêndios
    elif "COUNT_FOR_STD" in agg_per_hour.columns: 
        agg_per_hour["VALOR_STD"] = agg_per_hour.apply(lambda r: 0 if r.get("COUNT_FOR_STD", 0) <= 1 else r.get("VALOR_STD", 0), axis=1).fillna(0) # Std = 0 se contagem <= 1
    if "VALOR_STD" not in agg_per_hour: 
        agg_per_hour["VALOR_STD"] = 0.0 # Garante que VALOR_STD existe
    agg_per_hour["VALOR_STD"] = agg_per_hour["VALOR_STD"].fillna(0) # Preenche NaNs no STD com 0

    # Garante que todas as 24 horas estão presentes, preenchendo com 0 onde não há dados
    base_horas_df = pd.DataFrame({"HORA": range(24)})
    agg_hora_final = base_horas_df.merge(agg_per_hour, on="HORA", how="left").fillna(0)
    agg_hora_final["HORA_FMT"] = agg_hora_final["HORA"].apply(lambda h: f"{h:02d}h") # Formata hora para display (00h, 01h, ...)
    
    # Valores para o gráfico (média, limite superior/inferior da banda)
    y_mean = agg_hora_final.get("VALOR_MEAN", pd.Series(0.0, index=agg_hora_final.index))
    y_std = agg_hora_final.get("VALOR_STD", pd.Series(0.0, index=agg_hora_final.index))
    y_upper = (y_mean + y_std)
    y_lower = (y_mean - y_std).clip(lower=0) # Garante que o limite inferior não é negativo
    
    # Inicializa variáveis que dependem da métrica
    custom_data_scatter = None; hovertemplate = None; chart_title = ""; y_axis_title_text = ""; tickformat_yaxis = None
    y_plot_values = y_mean; y_upper_plot_values = y_upper; y_lower_plot_values = y_lower
    y_axis_range_effective_limit = y_upper_plot_values.max() * 1.3 if not y_upper_plot_values.empty else 0 # Define limite do eixo Y

    # Ajustes específicos para cada métrica
    if metric == "DURACAO_MEDIA":
        y_plot_values = y_mean / 60.0; y_upper_plot_values = y_upper / 60.0; y_lower_plot_values = y_lower / 60.0 # Converte para horas
        y_axis_range_effective_limit = max(1.0, y_upper_plot_values.max() * 1.3 if not y_upper_plot_values.empty else 1.0) # Ajusta limite do eixo Y
        agg_hora_final["VALOR_MEAN_DHM_STR"] = y_mean.apply(format_duration_dhm_verbose_refined) # Formata média para hover
        agg_hora_final["VALOR_STD_HM_STR"] = y_std.apply(format_duration_hm) # Formata std para hover
        custom_data_scatter = np.stack((agg_hora_final["VALOR_MEAN_DHM_STR"], agg_hora_final["VALOR_STD_HM_STR"]), axis=-1)
        trace_name = "Duração Média"; hovertemplate = "<b>Duração Média:</b> %{customdata[0]}<br><b>Desvio Padrão:</b> ±%{customdata[1]}<extra></extra>"
        chart_title = "Evolução Horária da Duração Média"; y_axis_title_text = "Duração Média (h)"; tickformat_yaxis = ',.0~f'
    elif metric == "AREA_ARDIDA":
        y_axis_range_effective_limit = max(10.0, y_upper_plot_values.max() * 1.3 if not y_upper_plot_values.empty else 10.0)
        trace_name = "Área Média (ha)"; hovertemplate = "<b>Área média (ha)</b>: %{y:,.1f}<br>Desvio padrão: ±%{customdata[0]:,.1f}<extra></extra>"
        chart_title = "Evolução Horária da Área Ardida Média"; y_axis_title_text = "área média (ha)"; custom_data_scatter = np.stack((y_std.round(1),), axis=-1); tickformat_yaxis = ',.0f'
    else: # NUM_INCENDIOS
        y_axis_range_effective_limit = max(10.0, y_upper_plot_values.max() * 1.3 if not y_upper_plot_values.empty else 10.0)
        trace_name = "Nº Incêndios"; hovertemplate = "<b>Nº incêndios</b>: %{y:.0f}<br>Faixa (±10%): %{customdata[0]:.0f} – %{customdata[1]:.0f}<extra></extra>"
        chart_title = "Evolução Horária do Nº de Incêndios"; y_axis_title_text = "Nº Incêndios"; custom_data_scatter = np.stack((y_lower.round(0), y_upper.round(0)), axis=-1); tickformat_yaxis = ',.0f'

    fig = go.Figure()
    # para a banda de variabilidade (área sombreada)
    fig.add_trace(go.Scatter(x=agg_hora_final["HORA_FMT"].tolist() + agg_hora_final["HORA_FMT"].tolist()[::-1], y=y_upper_plot_values.tolist() + y_lower_plot_values.tolist()[::-1], fill='toself', fillcolor='rgba(255, 182, 109, 0.3)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=agg_hora_final["HORA_FMT"],
        y=y_plot_values,
        mode='lines+markers', name=trace_name,
        line=dict(color=PALETTE.get("accent_red", "#E74C3C"), width=2),
        marker=dict(color=PALETTE.get("accent_orange", "#F39C12"), size=6, line=dict(width=1, color='white')),
        customdata=custom_data_scatter, # Dados para hover
        hovertemplate=hovertemplate # Template do hover
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=20, b=40), paper_bgcolor=PALETTE.get("card_bg", "#FFFFFF"), plot_bgcolor=PALETTE.get("card_bg", "#FFFFFF"), font=dict(color=PALETTE.get("font", "#2C3E50")),
                      xaxis=dict(title=None, tickangle=-45, showgrid=False, tickfont=dict(size=FONT_SIZE_TICK_LABEL), fixedrange=True), # Eixo X (Hora)
                      yaxis=dict(title=dict(text=y_axis_title_text, font=dict(size=FONT_SIZE_AXIS_TITLE)), showgrid=True, gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=FONT_SIZE_TICK_LABEL), range=[-0.02 * y_axis_range_effective_limit, y_axis_range_effective_limit]), # Eixo Y
                      hovermode="x unified", showlegend=False) # Hover unificado por X
    if tickformat_yaxis: 
        fig.update_layout(yaxis_tickformat=tickformat_yaxis, yaxis_nticks=5) # Formato e número de ticks do eixo Y
    else: 
        fig.update_layout(yaxis_tickmode='auto', yaxis_nticks=5)
    return fig

# Gráfico de Pizza para Tipos de Causa
def fig_pie_causas(df_chart_data: pd.DataFrame, metric: str, active_filter_name_for_title: str, ano: int, mes_val: int, height: int = 200) -> go.Figure:
    time_period = get_time_period_string(ano, mes_val)
    empty_msg_base = f"Sem dados de tipos de causa ({metric.lower()})<br>para {active_filter_name_for_title.lower()} ({time_period.lower()})."

    if df_chart_data.empty or "TIPOCAUSA" not in df_chart_data.columns:
        return create_empty_figure(empty_msg_base, height=height)

    df_causas_data = df_chart_data.copy()
    df_causas_data["TIPOCAUSA"] = df_causas_data["TIPOCAUSA"].fillna("NÃO ESPECIFICADA") # Preenche NaNs

    hover_label_metric = ""; hover_unit = ""; custom_data_column_names = [] # Para px.pie custom_data

    # Agrega dados com base na métrica selecionada
    if metric == "NUM_INCENDIOS":
        agg_data = df_causas_data["TIPOCAUSA"].value_counts().reset_index()
        agg_data.columns = ["TIPOCAUSA", "Valor"]
        hover_label_metric = "Nº Incêndios"
    elif metric == "AREA_ARDIDA":
        agg_data = df_causas_data.groupby("TIPOCAUSA")["AREATOTAL"].sum().reset_index(name="Valor")
        hover_label_metric = "Área Ardida"; hover_unit = " ha"
    elif metric == "DURACAO_MEDIA": # Para Duração, agregamos a soma e formatamos para hover
        df_causas_data['DURACAO_VALIDA'] = pd.to_numeric(df_causas_data['DURACAO'], errors='coerce').fillna(0)
        agg_data = df_causas_data.groupby("TIPOCAUSA")["DURACAO_VALIDA"].sum().reset_index(name="Valor")
        hover_label_metric = "Duração Total"
        agg_data["ValorFormatadoHM"] = agg_data["Valor"].apply(format_duration_hm)
        custom_data_column_names = ["ValorFormatadoHM"] # Nome da coluna para px.pie custom_data
    else: # Fallback para NUM_INCENDIOS
        agg_data = df_causas_data["TIPOCAUSA"].value_counts().reset_index()
        agg_data.columns = ["TIPOCAUSA", "Valor"]
        hover_label_metric = "Nº Incêndios"

    agg_data = agg_data[agg_data["Valor"] > 0] # Remove causas com valor 0
    if agg_data.empty: 
        return create_empty_figure(empty_msg_base, height=height)
    agg_data = agg_data.sort_values(by="Valor", ascending=False) # Ordena por valor

    if metric == "DURACAO_MEDIA":
        # %{customdata[0]} refere-se à primeira (e única) coluna em custom_data_column_names ('ValorFormatadoHM')
        hovertemplate_str = f"<b>%{{label}}</b><br>{hover_label_metric}: %{{customdata[0]}} (%{{percent}})<extra></extra>"
    else: # Para outras métricas, usa %{value}
        hovertemplate_str = f"<b>%{{label}}</b><br>{hover_label_metric}: %{{value:,.0f}}{hover_unit} (%{{percent}})<extra></extra>"

    # Cria o gráfico de pizza
    fig = px.pie(agg_data, names="TIPOCAUSA", values="Valor", hole=0.4, # hole para efeito "donut"
                 color="TIPOCAUSA", color_discrete_map=TIPOCAUSA_COLOR_MAP, # Cores personalizadas
                 custom_data=custom_data_column_names if custom_data_column_names else None) # CORREÇÃO: custom_data aqui

    fig.update_traces(
        textposition='outside', textinfo='label+percent', textfont_size=FONT_SIZE_PIE_TEXT, # Posição e info do texto nas fatias
        marker=dict(line=dict(color=PALETTE.get("card_bg", "#FFFFFF"), width=1.5)), # Linha de separação entre fatias
        pull=[0.05 if i == 0 else 0 for i in range(len(agg_data))], # "Puxa" a maior fatia
        insidetextorientation='horizontal',
        hovertemplate=hovertemplate_str # Template do hover
    )
    fig.update_layout(
        title=None, margin=dict(l=20, r=20, t=25, b=20), font=dict(color=PALETTE["font"]),
        showlegend=False, paper_bgcolor=PALETTE["card_bg"], plot_bgcolor=PALETTE["card_bg"],
        uniformtext_minsize=FONT_SIZE_PIE_TEXT -1, uniformtext_mode='hide', height=height # Ajuste de texto uniforme
    )
    return fig

# Nuvem de Palavras para Famílias de Causa
def img_nuvem_palavras(df_chart_data: pd.DataFrame, metric: str, active_filter_name_for_title: str, ano: int, mes_val: int, width: int = 400, height: int = 300):
    time_period = get_time_period_string(ano, mes_val)
    empty_msg_base = f"Sem dados de famílias de causa ({metric.lower()})\npara {active_filter_name_for_title.lower()} ({time_period.lower()})."
    if df_chart_data.empty or "CAUSAFAMILIA" not in df_chart_data.columns or df_chart_data["CAUSAFAMILIA"].dropna().empty:
        return empty_msg_base # Retorna mensagem se não houver dados

    df_nuvem_data = df_chart_data.copy()
    frequencies = {} # Dicionário para armazenar frequências/valores das famílias de causa
    
    # Calcula frequências/valores com base na métrica
    if metric == "NUM_INCENDIOS": 
        frequencies = df_nuvem_data["CAUSAFAMILIA"].dropna().value_counts().to_dict()
    elif metric == "AREA_ARDIDA": 
        frequencies = df_nuvem_data.groupby("CAUSAFAMILIA")["AREATOTAL"].sum().fillna(0).to_dict()
    elif metric == "DURACAO_MEDIA":
        df_nuvem_data['DURACAO_VALIDA'] = pd.to_numeric(df_nuvem_data['DURACAO'], errors='coerce').fillna(0)
        frequencies = df_nuvem_data.groupby("CAUSAFAMILIA")["DURACAO_VALIDA"].sum().fillna(0).to_dict()
    else: frequencies = df_nuvem_data["CAUSAFAMILIA"].dropna().value_counts().to_dict() # Fallback
    
    frequencies = {k: v for k, v in frequencies.items() if v > 0} # Remove famílias com valor 0
    if not frequencies: return empty_msg_base

    wc = WordCloud(width=width, height=height, background_color=PALETTE["card_bg"], color_func=wordcloud_color_func, # Usa a função de cor personalizada
                   max_font_size=WORDCLOUD_MAX_FONT_SIZE, min_font_size=WORDCLOUD_MIN_FONT_SIZE, min_word_length=1)
    try:
        wc.generate_from_frequencies(frequencies) 
    except ValueError as e: #
        return f"Erro ao gerar nuvem de palavras: {e}.\nVerifique os dados de frequência para\n{active_filter_name_for_title.lower()} ({time_period.lower()})."
    
    # Converte a imagem da nuvem para formato base64 para embutir em HTML
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# Gráfico de Violino para Distribuição de Variáveis Meteorológicas
def fig_violin_distribution(df_chart_data: pd.DataFrame, active_filter_name_for_title: str, ano: int, mes_val: int, height: int = 305):
    time_period = get_time_period_string(ano, mes_val)
    empty_msg_base = f"Sem dados meteorológicos para {active_filter_name_for_title.lower()}<br>({time_period.lower()})."
    if df_chart_data.empty: 
        return create_empty_figure(empty_msg_base, height=height)
    
    meteo_cols = ["TEMPERATURA", "HUMIDADERELATIVA", "VENTOINTENSIDADE"] # Colunas relevantes
    df_meteo = df_chart_data[meteo_cols].dropna(how='all') # Remove linhas onde todas estas colunas são NaN
    if df_meteo.empty: 
        return create_empty_figure(empty_msg_base, height=height)
    
    # Transforma o DataFrame para formato "longo" (ideal para px.violin)
    melted_df = df_meteo.melt(var_name="ParametroTecnico", value_name="ValorMedido").dropna()
    melted_df["ParametroDisplay"] = melted_df["ParametroTecnico"].map(METEO_LABELS_MAP) # Mapeia para nomes de display
    if melted_df.empty or melted_df["ParametroDisplay"].isnull().all(): 
        return create_empty_figure(empty_msg_base, height=height)
    
    # Define a ordem e cores dos violinos
    valid_ordered_params = [METEO_LABELS_MAP.get(p) for p in ["TEMPERATURA", "HUMIDADERELATIVA", "VENTOINTENSIDADE"] if METEO_LABELS_MAP.get(p) is not None]
    color_map = {METEO_LABELS_MAP.get(k,k):v for k,v in {"TEMPERATURA":PALETTE["accent_red"], "HUMIDADERELATIVA":PALETTE["accent_orange"], "VENTOINTENSIDADE":PALETTE["brand"]}.items()}
    
    # Calcula estatísticas para o hover
    hover_stats = {}
    for param in valid_ordered_params:
        param_data = melted_df[melted_df["ParametroDisplay"] == param]["ValorMedido"]
        if not param_data.empty: 
            hover_stats[param] = {"min": round(param_data.min()), "max": round(param_data.max()), "media": round(param_data.mean()), "mediana": round(param_data.median())}
    
    # Cria o gráfico de violino
    fig = px.violin(melted_df, x="ParametroDisplay", y="ValorMedido", box=True, points=False, color="ParametroDisplay", # box=True mostra boxplot interno
                    color_discrete_map=color_map, category_orders={"ParametroDisplay": valid_ordered_params}, # Ordem e cores
                    labels={"ParametroDisplay":"Parâmetro", "ValorMedido":"Valor"})
    
    # Personaliza o hovertemplate para cada violino
    for i, param in enumerate(valid_ordered_params):
        if param in hover_stats:
            stats = hover_stats[param]
            fig.data[i].hovertemplate = f"<b>{param}</b><br>Média: {stats['media']}<br>Mediana: {stats['mediana']}<br>Mínimo: {stats['min']}<br>Máximo: {stats['max']}<extra></extra>"
    
    fig.update_traces(meanline_visible=True, width=0.6, marker=dict(opacity=0.3), line=dict(width=1.2)) # Mostra linha da média
    fig.update_layout(
        height=height, paper_bgcolor=PALETTE["card_bg"], plot_bgcolor=PALETTE["card_bg"], font_color=PALETTE["font"],
        margin=dict(l=20,r=20,t=20,b=20), showlegend=False,
        xaxis=dict(title=None, tickfont=dict(size=FONT_SIZE_TICK_LABEL), fixedrange=True), # Eixo X (Parâmetros)
        yaxis=dict(title=dict(text="Valor Registado", font=dict(size=FONT_SIZE_AXIS_TITLE)), tickfont=dict(size=FONT_SIZE_TICK_LABEL), showgrid=True, gridcolor="rgba(0,0,0,0.05)"), # Eixo Y (Valores)
        hovermode='closest', hoverlabel=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["font"], font=dict(size=12, color=PALETTE["font"]), align="left")
    )
    return fig

# Gráfico de Dispersão: Temperatura vs Humidade, com tamanho dos pontos pela Intensidade do Vento
def fig_scatter_meteo(df_chart_data: pd.DataFrame, active_filter_name_for_title: str, ano: int, mes_val: int, height: int = 305) -> go.Figure:
    time_period_str = get_time_period_string(ano, mes_val)
    base_error_msg = f"Sem dados de Temp/Hum/Vento (Agrícola/Florestal)<br>para {active_filter_name_for_title.lower()} ({time_period_str.lower()})."
    if df_chart_data.empty: return create_empty_figure(base_error_msg, height=height)
    
    meteo_cols = ["TEMPERATURA", "HUMIDADERELATIVA", "VENTOINTENSIDADE", "TIPO"] # Colunas necessárias
    df_to_filter = df_chart_data.copy()
    df_to_filter['VENTOINTENSIDADE'] = pd.to_numeric(df_to_filter['VENTOINTENSIDADE'], errors='coerce') # Converte Vento para numérico
    df_meteo_tipo = df_to_filter[meteo_cols].dropna(subset=[col for col in meteo_cols if col != 'TIPO']) # Remove NaNs (exceto em TIPO)
    # Filtra apenas para tipos Agrícola e Florestal e vento não negativo
    df_scatter = df_meteo_tipo[df_meteo_tipo["TIPO"].isin(["Agrícola", "Florestal"]) & (df_meteo_tipo["VENTOINTENSIDADE"] >= 0)].copy()
    if df_scatter.empty: 
        return create_empty_figure(base_error_msg, height=height)
    
    present_tipos = df_scatter["TIPO"].unique()
    current_color_map = {tipo: COLOR_MAP_TIPO[tipo] for tipo in present_tipos if tipo in COLOR_MAP_TIPO}
    if not current_color_map: 
        return create_empty_figure(f"Tipos 'Agrícola' ou 'Florestal' não encontrados<br>ou sem mapeamento de cor. {base_error_msg}", height=height)
    
    size_max_dynamic = 15 # Tamanho max dos pontos
    
    fig = px.scatter(
        df_scatter, x="TEMPERATURA", y="HUMIDADERELATIVA", color="TIPO", symbol="TIPO", # Eixos, cor e símbolo
        symbol_map={"Florestal": "circle", "Agrícola": "circle"}, # Mesmo símbolo para ambos os tipos
        size="VENTOINTENSIDADE", size_max=size_max_dynamic, # Tamanho dos pontos pelo Vento
        labels={"TEMPERATURA": "Temperatura (°C)", "HUMIDADERELATIVA": "Humidade Relativa (%)", "TIPO": "tipo", "VENTOINTENSIDADE": "Vento (km/h)"}, # Labels dos eixos
        color_discrete_map=current_color_map, custom_data=['VENTOINTENSIDADE'], # Cores e customdata para hover
    )
    fig.update_traces(
        marker=dict(opacity=0.8, line=dict(width=0, color='rgba(0,0,0,0)')), selector=dict(type='scatter'),
        hovertemplate="<b>tipo: %{fullData.name}</b><br>temp: %{x:.1f}°c<br>hum: %{y:.1f}%<br>vento: %{customdata[0]:.1f} km/h<br><b>Tamanho por Vento</b><extra></extra>" # Template do hover
    )
    fig.update_layout(
        title=None, height=height, paper_bgcolor=PALETTE.get("card_bg", "#FFFFFF"), plot_bgcolor=PALETTE.get("card_bg", "#FFFFFF"),
        font_color=PALETTE.get("font", "#2C3E50"), margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title=dict(text="Temperatura (°C)", font=dict(size=FONT_SIZE_AXIS_TITLE)), tickfont=dict(size=FONT_SIZE_TICK_LABEL), showgrid=True, gridcolor="rgba(0,0,0,0.05)"), # Eixo X
        yaxis=dict(title=dict(text="Humidade Relativa (%)", font=dict(size=FONT_SIZE_AXIS_TITLE)), tickfont=dict(size=FONT_SIZE_TICK_LABEL), showgrid=True, gridcolor="rgba(0,0,0,0.05)"), # Eixo Y
        legend=dict( # Legenda
            title=dict(text="Tipo", font=dict(size=FONT_SIZE_LEGEND_TITLE)), font=dict(size=FONT_SIZE_LEGEND_ITEM),
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, itemsizing='constant', bgcolor='rgba(255,255,255,0.7)'
        )
    )
    return fig

# Calcula a vista do mapa (centro e zoom) com base nos dados e filtros selecionados
def _calculate_map_view(df_geo: pd.DataFrame, sel_dist: str, sel_conc: str):
    if df_geo.empty or 'LAT' not in df_geo.columns or 'LON' not in df_geo.columns: 
        return {"center": DEFAULT_CENTER_PT, "zoom": ZOOM_PORTUGAL} # Default se não houver dados
    
    # Calcula o centro como a média das latitudes e longitudes
    center_lat = df_geo["LAT"].mean(); center_lon = df_geo["LON"].mean()
    if pd.isna(center_lat) or pd.isna(center_lon): 
        return {"center": DEFAULT_CENTER_PT, "zoom": ZOOM_PORTUGAL} # Default se médias forem NaN
    
    view = {"center": {"lat": center_lat, "lon": center_lon}}
    
    # Ajusta o zoom com base na seleção de concelho ou distrito
    if sel_conc != "Todos": # Se um concelho está selecionado
        lat_min, lat_max = df_geo["LAT"].min(), df_geo["LAT"].max()
        lon_min, lon_max = df_geo["LON"].min(), df_geo["LON"].max()
        if pd.notna(lat_min) and pd.notna(lat_max) and pd.notna(lon_min) and pd.notna(lon_max):
            # Calcula o zoom com base na extensão geográfica dos dados do concelho
            lat_span = lat_max - lat_min; lon_span = lon_max - lon_min; max_span = max(lat_span, lon_span)
            if max_span < 0.001: 
                zoom = 14
            elif max_span < 0.05: 
                zoom = 12
            elif max_span < 0.15: 
                zoom = 11
            elif max_span < 0.4: 
                zoom = 10
            elif max_span < 1.0: 
                zoom = 9
            else: zoom = 8
            view["zoom"] = min(zoom, MAX_ZOOM_CONCELHO) # Limita ao zoom máximo para concelho
        else: 
            view["zoom"] = ZOOM_DISTRITO # Fallback para zoom de distrito
    elif sel_dist != "Todos": 
        view["zoom"] = ZOOM_DISTRITO # Zoom de distrito
    else: # Nenhum filtro geográfico específico (Portugal Continental)
        view["center"] = DEFAULT_CENTER_PT; view["zoom"] = ZOOM_PORTUGAL
    return view

# Cria um mapa de dispersão como fallback para mapas meteorológicos de densidade quando há poucos pontos
def _create_meteo_scatter_fallback_map(df_display, variable, var_label, color_scale, range_color, sel_dist, sel_conc, palette_dict, fig_height):
    map_view = _calculate_map_view(df_display, sel_dist, sel_conc) # Calcula vista do mapa
    
    if df_display.empty or not all(col in df_display.columns for col in ['LAT', 'LON', variable, 'CONCELHO']):
         return create_empty_figure(f"Dados insuficientes de {var_label.lower()}<br>para exibir no mapa de pontos.", height=fig_height)
    
    df_scatter = df_display.dropna(subset=['LAT', 'LON', variable, 'CONCELHO']).copy() # Remove NaNs
    if df_scatter.empty: return create_empty_figure(f"Sem dados válidos de {var_label.lower()}<br>para exibir no mapa de pontos.", height=fig_height)
    
    fig = px.scatter_mapbox(df_scatter, lat='LAT', lon='LON', color=variable, size_max=8, 
                            center=map_view["center"], zoom=map_view["zoom"], mapbox_style="carto-positron", 
                            labels={variable: f'{var_label}'}, color_continuous_scale=color_scale, 
                            opacity=0.85, height=fig_height, range_color=range_color, 
                            hover_name='CONCELHO', custom_data=[variable])
    fig.update_traces(marker=dict(size=7), hovertemplate="<b>%{hovertext}</b><br>" + f"{var_label}: %{{customdata[0]:.1f}}<extra></extra>") # Hover
    fig.update_layout(
        dragmode="zoom", margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict( # Colorbar
            title=dict(text=f'{var_label}', font=dict(size=FONT_SIZE_COLORBAR_TITLE)),
            ticksuffix='°c' if variable == 'TEMPERATURA' else '%' if variable == 'HUMIDADERELATIVA' else '', # Sufixo da unidade
            orientation="h", y=0.98, yanchor="bottom", len=0.8, x=0.5, xanchor='center',
            thickness=15, tickfont=dict(size=FONT_SIZE_COLORBAR_TICK)
        ),
        paper_bgcolor=palette_dict.get("card_bg", "#FFFFFF"), plot_bgcolor=palette_dict.get("card_bg", "#FFFFFF"),
        font=dict(family="Arial", size=12, color=palette_dict.get("font", "#2C3E50")), sliders=None, updatemenus=None # Sem controlos de animação
    )
    return fig

# Calcula as coordenadas para uma forma de seta (corpo e cabeça)
def create_arrow_shape(lat, lon, angle_rad, intensity_val, scale=0.08, intensity_normalization_cap=30.0, min_length_factor=0.2, head_connection_scale_factor=0.6, head_wing_size_factor=0.45):
    # Normaliza a intensidade do vento para o comprimento da seta
    if intensity_normalization_cap <= 0: normalized_intensity_for_length = 0.5
    else: normalized_intensity_for_length = min(1.0, intensity_val / intensity_normalization_cap)
    current_length = scale * (min_length_factor + (1 - min_length_factor) * normalized_intensity_for_length) # Comprimento da seta
    
    base_lat, base_lon = lat, lon
    tip_lat = lat + current_length * np.sin(angle_rad)
    tip_lon = lon + current_length * np.cos(angle_rad)
    
    head_attach_lat = lat + head_connection_scale_factor * current_length * np.sin(angle_rad)
    head_attach_lon = lon + head_connection_scale_factor * current_length * np.cos(angle_rad)
    wing_length = current_length * head_wing_size_factor
    wing_angle_offset = np.pi * 3/4 # Ângulo das "asas" da cabeça
    wing1_lat = head_attach_lat + wing_length * np.sin(angle_rad + wing_angle_offset)
    wing1_lon = head_attach_lon + wing_length * np.cos(angle_rad + wing_angle_offset)
    wing2_lat = head_attach_lat + wing_length * np.sin(angle_rad - wing_angle_offset)
    wing2_lon = head_attach_lon + wing_length * np.cos(angle_rad - wing_angle_offset)
    
    shaft_lats = [base_lat, tip_lat]; shaft_lons = [base_lon, tip_lon] # Corpo da seta
    head_triangle_lats = [wing1_lat, tip_lat, wing2_lat]; head_triangle_lons = [wing1_lon, tip_lon, wing2_lon] # Cabeça da seta
    return shaft_lats, shaft_lons, head_triangle_lats, head_triangle_lons

# Obtém a cor da seta com base na intensidade do vento
def get_arrow_color_from_intensity(intensity_val, cmin=0, cmax=40, colorscale_name=WIND_COLOR_SCALE):
    if pd.isna(intensity_val): return 'rgba(0,0,0,0.1)' # Cor para valor NaN
    # Normaliza o valor da intensidade para a escala de cores
    if cmin == cmax: 
        norm_val = 0.5
    elif intensity_val <= cmin: 
        norm_val = 0.0
    elif intensity_val >= cmax: 
        norm_val = 1.0
    else: 
        norm_val = (intensity_val - cmin) / (cmax - cmin)
    return px.colors.sample_colorscale(colorscale_name, [norm_val])[0]

def add_arrow_traces(df_rows, fig_or_list):
    for _, row in df_rows.iterrows(): # Itera sobre as linhas de dados (cada linha é uma seta)
        angle = np.radians(row["PLOTLY_ANGLE"]) # Ângulo da seta
        # Cria a forma da seta
        shaft_lat, shaft_lon, head_lat, head_lon = create_arrow_shape(
            row["LAT"], row["LON"], angle, row["VENTOINTENSIDADE"],
            scale=BASE_SCALE, intensity_normalization_cap=LENGTH_CAP,
            min_length_factor=0.2, head_connection_scale_factor=HEAD_CONN, head_wing_size_factor=HEAD_WING
        )
        color = row["ARROW_COLOR_VAL"] # Cor da seta
        # Trace para o corpo da seta
        shaft_trace = go.Scattermapbox(lat=shaft_lat, lon=shaft_lon, mode="lines", line=dict(width=SHAFT_WIDTH, color=color),
                                     hoverinfo="text", hovertext=f"<b>{row['CONCELHO']}</b><br>Vento: {row['VENTOINTENSIDADE']:.1f} km/h", showlegend=False)
        # Trace para a cabeça da seta (preenchida)
        head_trace = go.Scattermapbox(lat=head_lat, lon=head_lon, mode="lines", line=dict(width=HEAD_WIDTH, color=color),
                                    fill="toself", fillcolor=color, hoverinfo="text",
                                    hovertext=f"<b>{row['CONCELHO']}</b><br>Vento: {row['VENTOINTENSIDADE']:.1f} km/h", showlegend=False)
        # Adiciona os traces à figura ou lista
        if isinstance(fig_or_list, go.Figure):
            fig_or_list.add_trace(shaft_trace); fig_or_list.add_trace(head_trace)
        else:
            fig_or_list.append(shaft_trace); fig_or_list.append(head_trace)
# --- Fim das Funções de Setas de Vento (Não Usadas Atualmente) ---


# Cria o mapa de densidade para Humidade Relativa, com animação por dia se disponível
def _create_humidity_density_map(df_display, ano, mes_val, sel_dist, sel_conc, palette_dict, meses_ext_dict, fig_height):
    mes_nome = meses_ext_dict.get(mes_val, f"mês {mes_val}") if mes_val != 0 else "todo o ano"
    local_str = f" ({sel_conc.lower()})" if sel_conc != 'Todos' else f" ({sel_dist.lower()})" if sel_dist != 'Todos' else ""
    empty_msg = f"Sem dados de humidade válidos para exibir<br>({mes_nome.lower()} de {ano}{local_str})"
    
    required_cols_density = ['LAT', 'LON', 'HUMIDADERELATIVA', 'CONCELHO'] # Colunas necessárias
    if df_display.empty or not all(c in df_display.columns for c in required_cols_density): 
        return create_empty_figure(empty_msg + "<br>(colunas em falta)", height=fig_height)

    df_valid_hum = df_display.dropna(subset=required_cols_density) # Remove NaNs
    if df_valid_hum.empty: 
        return create_empty_figure(empty_msg, height=fig_height)
    
    # Se houver muito poucos pontos (1 ou 2), usa um mapa de dispersão (scatter) como fallback
    if 1 <= len(df_valid_hum) < 3: 
        return _create_meteo_scatter_fallback_map(df_valid_hum, variable='HUMIDADERELATIVA', var_label='Humidade (%)', color_scale="Blues", range_color=[0, 100], sel_dist=sel_dist, sel_conc=sel_conc, palette_dict=palette_dict, fig_height=fig_height)
    
    map_view = _calculate_map_view(df_valid_hum, sel_dist, sel_conc) # Calcula centro e zoom
    
    # Prepara dados para animação por dia, se a coluna 'DIA' existir e tiver dados válidos
    if 'DIA' not in df_valid_hum.columns or df_valid_hum['DIA'].isnull().all():
        df_anim = df_valid_hum.copy(); df_anim['DIA'] = 1 # Assume Dia 1 se não houver dados de dia
        unique_dias = [1]
    else:
        df_anim = df_valid_hum.dropna(subset=['DIA']).copy()
        if df_anim.empty : return create_empty_figure(f"Sem dados de 'dia' para animação da humidade<br>{empty_msg}", height=fig_height)
        df_anim['DIA'] = df_anim['DIA'].astype(int) # Converte Dia para inteiro
        unique_dias = sorted(df_anim['DIA'].unique()) # Dias únicos para os frames da animação
    
    if df_anim.empty: return create_empty_figure(f"Sem dados para animação da humidade<br>{empty_msg}", height=fig_height)
    
    use_animation = len(unique_dias) > 1 # Animação só se houver mais de um dia
    
    mapbox_args = dict(
        lat='LAT', lon='LON', z='HUMIDADERELATIVA', radius=26, # z é a variável para a densidade/cor
        center=map_view["center"], zoom=map_view["zoom"], 
        mapbox_style="carto-positron", 
        labels={'HUMIDADERELATIVA': 'Humidade (%)'}, 
        color_continuous_scale="Blues", opacity=1, height=fig_height, range_color=[0, 100], # Escala "Blues"
        custom_data=['CONCELHO'] 
    )
    if use_animation:
        mapbox_args['animation_frame'] = 'DIA'
        mapbox_args['category_orders'] = {"DIA": unique_dias} # Garante a ordem correta dos frames
        
    fig = px.density_mapbox(df_anim, **mapbox_args) # Cria o mapa de densidade
    
    fig.update_traces(hovertemplate="<b>Concelho: %{customdata[0]}</b><br>Humidade: %{z:.1f}%<extra></extra>") # Template do hover
    
    fig.update_layout(
        dragmode="zoom", margin=dict(l=0, r=0, t=30, b=70 if use_animation else 5), # Margem inferiooor maior se houver animação (para controlos)
        coloraxis_colorbar=dict( # Colorbar
            title=dict(text='Humidade (%)', font=dict(size=FONT_SIZE_COLORBAR_TITLE)), ticksuffix='%',
            orientation="h", y=0.98, yanchor="bottom", len=0.8, x=0.5, xanchor='center',
            thickness=15, tickfont=dict(size=FONT_SIZE_COLORBAR_TICK)
        ),
        paper_bgcolor=palette_dict.get("card_bg", "#FFFFFF"), plot_bgcolor=palette_dict.get("card_bg", "#FFFFFF"),
        font=dict(family="Arial", size=12, color=palette_dict.get("font", "#2C3E50")), mapbox_domain={'y': [0, 0.975]} # Domínio do mapa para dar espaço à colorbar
    )
    if use_animation: # config controlos de animação (slider e botões play/pause) !
        fig.update_layout(sliders=[{'active': 0, 'yanchor': 'top', 'xanchor': 'left',
                                     'currentvalue': {'font': {'size': FONT_SIZE_ANIMATION_VALUE, 'color': palette_dict.get("font")}, 'prefix': 'Dia: ', 'visible': True, 'xanchor': 'left'},
                                     'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 0, 't': 10},
                                     'len': 0.75, 'x': 0.1, 'y': 0.05, 'bgcolor': 'rgba(200,200,200,0.3)',
                                     'activebgcolor': palette_dict.get("brand"), 'borderwidth': 0,
                                     'steps': [{'args': [[f'{day}'], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}], 'label': str(day), 'method': 'animate'} for day in unique_dias]}],
                          updatemenus=[dict(type="buttons", direction="left", showactive=True,
                                             bgcolor=palette_dict.get("card_bg", "#EEEEEE"), bordercolor="#CCCCCC", borderwidth=1,
                                             font={'size': FONT_SIZE_ANIMATION_BUTTON, 'color': palette_dict.get("font")},
                                             buttons=[dict(label="▶", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]),
                                                      dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])],
                                             pad={"r": 10, "l": 5, "t": 0, "b": 0}, x=0.95, xanchor="right", y=0, yanchor="bottom")])
    else: fig.update_layout(sliders=None, updatemenus=None) # Remove controlos se não houver animação
    return fig

# Cria o mapa de densidade para Temperatura, com animação por dia se disponível
def _create_temperature_density_map(df_display, ano, mes_val, sel_dist, sel_conc, palette_dict, meses_ext_dict, fig_height):
    # Lógica muito semelhante a _create_humidity_density_map, mas para Temperatura
    mes_nome = meses_ext_dict.get(mes_val, f"mês {mes_val}") if mes_val != 0 else "todo o ano"
    local_str = f" ({sel_conc.lower()})" if sel_conc != 'Todos' else f" ({sel_dist.lower()})" if sel_dist != 'Todos' else ""
    empty_msg = f"Sem dados de temperatura válidos para exibir<br>({mes_nome.lower()} de {ano}{local_str})"
    
    required_cols_density = ['LAT', 'LON', 'TEMPERATURA', 'CONCELHO']
    if df_display.empty or not all(c in df_display.columns for c in required_cols_density): 
        return create_empty_figure(empty_msg + "<br>(colunas em falta)", height=fig_height)
        
    df_valid_temp = df_display.dropna(subset=required_cols_density)
    if df_valid_temp.empty: 
        return create_empty_figure(empty_msg, height=fig_height)
    
    # Fallback para scatter map se poucos pontos
    if 1 <= len(df_valid_temp) < 3: 
        return _create_meteo_scatter_fallback_map(df_valid_temp, variable='TEMPERATURA', var_label='Temperatura (°C)', color_scale=TEMP_COLOR_SCALE_DENSITY, range_color=TEMP_RANGE_DENSITY, sel_dist=sel_dist, sel_conc=sel_conc, palette_dict=palette_dict, fig_height=fig_height)
    
    map_view = _calculate_map_view(df_valid_temp, sel_dist, sel_conc)
    
    # Preparação para animação por dia
    if 'DIA' not in df_valid_temp.columns or df_valid_temp['DIA'].isnull().all():
        df_anim = df_valid_temp.copy(); df_anim['DIA'] = 1
        unique_dias = [1]
    else:
        df_anim = df_valid_temp.dropna(subset=['DIA']).copy()
        if df_anim.empty : return create_empty_figure(f"Sem dados de 'dia' para animação da temperatura<br>{empty_msg}", height=fig_height)
        df_anim['DIA'] = df_anim['DIA'].astype(int)
        unique_dias = sorted(df_anim['DIA'].unique())
        
    if df_anim.empty: return create_empty_figure(f"Sem dados para animação da temperatura<br>{empty_msg}", height=fig_height)
    
    use_animation = len(unique_dias) > 1
    # Ajusta o domínio Y do mapa para dar espaço aos controlos de animação, se ativos
    mapbox_domain_y = [0, 0.9] if use_animation else [0, 0.975] 
    
    # Argumentos para px.density_mapbox
    mapbox_args = dict(
        lat='LAT', lon='LON', z='TEMPERATURA', radius=26, 
        center=map_view["center"], zoom=map_view["zoom"], 
        mapbox_style="carto-positron", 
        labels={'TEMPERATURA': 'Temp (°C)'}, 
        color_continuous_scale=TEMP_COLOR_SCALE_DENSITY, opacity=1, height=fig_height, range_color=TEMP_RANGE_DENSITY, # Escala de cor para temperatura
        custom_data=['CONCELHO']
    )
    if use_animation: 
        mapbox_args['animation_frame'] = 'DIA'
        mapbox_args['category_orders'] = {"DIA": unique_dias}
        
    fig = px.density_mapbox(df_anim, **mapbox_args) # Cria o mapa
    
    fig.update_traces(hovertemplate="<b>Concelho: %{customdata[0]}</b><br>Temperatura: %{z:.1f}°C<extra></extra>") # Hover
    
    fig.update_layout(
        dragmode="zoom", margin=dict(l=0, r=0, t=30, b=70 if use_animation else 5),
        coloraxis_colorbar=dict( # Colorbar
            title=dict(text='Temperatura (°C)', font=dict(size=FONT_SIZE_COLORBAR_TITLE)), ticksuffix='°C',
            orientation="h", y=0.98, yanchor="bottom", len=0.8, x=0.5, xanchor='center',
            thickness=15, tickfont=dict(size=FONT_SIZE_COLORBAR_TICK)
        ),
        paper_bgcolor=palette_dict.get("card_bg", "#FFFFFF"), plot_bgcolor=palette_dict.get("card_bg", "#FFFFFF"),
        font=dict(family="Arial", size=12, color=palette_dict.get("font", "#2C3E50")), mapbox_domain={'y': mapbox_domain_y}
    )
    if use_animation: # Configura controlos de animação
         fig.update_layout(sliders=[{'active': 0, 'yanchor': 'top', 'xanchor': 'left',
                                     'currentvalue': {'font': {'size': FONT_SIZE_ANIMATION_VALUE, 'color': palette_dict.get("font")}, 'prefix': 'Dia: ', 'visible': True, 'xanchor': 'left'},
                                     'transition': {'duration': 300, 'easing': 'cubic-in-out'}, 'pad': {'b': 0, 't': 10},
                                     'len': 0.75, 'x': 0.1, 'y': 0.05, 'bgcolor': 'rgba(200,200,200,0.3)',
                                     'activebgcolor': palette_dict.get("brand"), 'borderwidth': 0,
                                     'steps': [{'args': [[f'{day}'], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}], 'label': str(day), 'method': 'animate'} for day in unique_dias]}],
                           updatemenus=[dict(type="buttons", direction="left", showactive=True,
                                             bgcolor=palette_dict.get("card_bg", "#EEEEEE"), bordercolor="#CCCCCC", borderwidth=1,
                                             font={'size': FONT_SIZE_ANIMATION_BUTTON, 'color': palette_dict.get("font")},
                                             buttons=[dict(label="▶", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "cubic-in-out"}}]),
                                                      dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])],
                                             pad={"r": 10, "l": 5, "t": 0, "b": 0}, x=0.95, xanchor="right", y=0.05, yanchor="top")]) # y=0.05 e yanchor="top" para alinhar com o slider
    else: fig.update_layout(sliders=None, updatemenus=None)
    return fig

# Função principal para criar o mapa meteorológico (Temperatura, Humidade ou Vento)
def fig_meteo_map(df_main_complete: pd.DataFrame, variable: str, selected_distrito: str, selected_concelho: str, ano: int, mes_val: int) -> go.Figure:
    meteo_map_height_numeric = int(METEO_MAP_NEW_HEIGHT.replace('px', '')) # Altura do mapa

    if df_main_complete.empty:
        return create_empty_figure("Dados principais não disponíveis.", height=meteo_map_height_numeric)

    # Filtra dados pelo ano e mês selecionados
    df_temp = df_main_complete[df_main_complete["ANO"] == ano]
    if mes_val != 0: # Se um mês específico foi selecionado
        df_temp = df_temp[df_temp["MES"] == mes_val]
    else: # Se "Todos os Meses" foi selecionado, mostra mensagem para escolher um mês
        return create_empty_figure("⚠️<br>Seleciona um mês específico<br>para ver o mapa meteorológico.", height=meteo_map_height_numeric)

    df_filtered_geo = df_temp.copy()
    if 'CONCELHO' not in df_filtered_geo.columns: # Garante que a coluna CONCELHO existe (para hover)
        df_filtered_geo['CONCELHO'] = "Desconhecido"
        
    # Filtra por distrito ou concelho, se selecionado
    if selected_concelho != "Todos":
        df_filtered_geo = df_filtered_geo[df_filtered_geo["CONCELHO"] == selected_concelho]
    elif selected_distrito != "Todos":
        df_filtered_geo = df_filtered_geo[df_filtered_geo["DISTRITO"] == selected_distrito]

    # Lógica específica para o mapa de VENTO
    if variable == "VENTOINTENSIDADE":
        required_cols = ["LAT", "LON", "VENTOINTENSIDADE", "VENTODIRECAO_VETOR", "CONCELHO"] # Colunas necessárias para vento
        has_dia_data_for_animation = "DIA" in df_filtered_geo.columns and not df_filtered_geo["DIA"].isnull().all() # Verifica se há dados de dia para animação

        df_wind = df_filtered_geo.dropna(subset=required_cols).copy() # Remove NaNs
        
        # Prepara dados de dia para animação
        if has_dia_data_for_animation:
            df_wind = df_wind.dropna(subset=["DIA"]).copy()
            if not df_wind.empty:
                 df_wind["DIA"] = pd.to_numeric(df_wind["DIA"], errors="coerce").dropna().astype(int)
            else: # Se ficou vazio após remover NaNs de DIA
                has_dia_data_for_animation = False 
                if not df_wind.empty: df_wind["DIA"] = 1 # Fallback para Dia 1
                else: return create_empty_figure("Sem dados de vento válidos com dia.", height=meteo_map_height_numeric)
        else: # Sem dados de DIA
            if not df_wind.empty: df_wind["DIA"] = 1 # Assume Dia 1
            has_dia_data_for_animation = False # Garante que animação não será usada

        if df_wind.empty:
            return create_empty_figure("Sem dados completos de vento<br>para este local e mês.", height=meteo_map_height_numeric)

        map_view = _calculate_map_view(df_wind, selected_distrito, selected_concelho) # Calcula centro e zoom
        
        unique_dias = sorted(df_wind["DIA"].unique()) if "DIA" in df_wind and not df_wind["DIA"].empty else [1] # Dias únicos para animação
        use_animation = has_dia_data_for_animation and len(unique_dias) > 1 # Animação se houver múltiplos dias
        
        # Colunas para customdata no hover do mapa de vento
        custom_data_for_wind = ["VENTOINTENSIDADE", "VENTODIRECAO_VETOR", "CONCELHO"]

        fig = px.density_mapbox(
            df_wind, lat="LAT", lon="LON", z="VENTOINTENSIDADE", radius=26,
            center=map_view["center"], zoom=map_view["zoom"], mapbox_style="carto-positron",
            labels={"VENTOINTENSIDADE": "Vento (km/h)"}, 
            color_continuous_scale=WIND_COLOR_SCALE, range_color=WIND_RANGE_COLOR, # Escala de cor para vento
            opacity=0.8, height=meteo_map_height_numeric,
            animation_frame="DIA" if use_animation else None, # Animação por DIA
            category_orders={"DIA": unique_dias} if use_animation else None,
            custom_data=custom_data_for_wind # Passa a lista de nomes de colunas para custom_data
        )

        fig.update_traces( # Template do hover para o mapa de densidade de vento
            hovertemplate="<b>Concelho: %{customdata[2]}</b><br>Vento: %{z:.1f} km/h<br>Direção: %{customdata[1]:.0f}°<extra></extra>"
        )
        
        fig.update_layout( # Layout geral do mapa de vento
            autosize=True, margin=dict(l=0, r=0, t=30, b=70 if use_animation else 5),
            dragmode="zoom", paper_bgcolor=PALETTE["card_bg"], font_color=PALETTE["font"],
            mapbox_domain={'y': [0, 0.975]}, # Espaço para colorbar
            coloraxis_colorbar=dict( # Colorbar para intensidade do vento
                title=dict(text="Vento (km/h)", font=dict(size=FONT_SIZE_COLORBAR_TITLE)),
                tickvals=[0, 10, 20, 30, WIND_RANGE_COLOR[1]], # Ticks personalizados
                ticktext=[str(t) for t in [0, 10, 20, 30, WIND_RANGE_COLOR[1]]],
                orientation="h", y=0.98, yanchor="bottom",
                x=0.5, xanchor="center", len=0.8, thickness=15,
                tickfont=dict(size=FONT_SIZE_COLORBAR_TICK)
            )
        )

        if use_animation: # Adiciona controlos de animação se aplicável
            fig.update_layout(
                sliders=[{ # Slider de dias
                    'active': 0, 'yanchor': 'top', 'xanchor': 'left',
                    'currentvalue': {'font': {'size': FONT_SIZE_ANIMATION_VALUE, 'color': PALETTE.get("font")}, 'prefix': 'Dia: ', 'visible': True, 'xanchor': 'left'},
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'pad': {'b': 0, 't': 10}, 'len': 0.75, 'x': 0.1, 'y': 0.05, 
                    'bgcolor': 'rgba(200,200,200,0.3)', 'activebgcolor': PALETTE.get("brand"),
                    'borderwidth': 0,
                    'steps': [{'args': [[f'{day}'], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}], 'label': str(day), 'method': 'animate'} for day in unique_dias]
                }],
                updatemenus=[{ # Botões Play/Pause
                    'type': "buttons", 'direction': "left", 'showactive': True,
                    'bgcolor': PALETTE.get("card_bg", "#EEEEEE"), 'bordercolor': "#CCCCCC", 'borderwidth': 1,
                    'font': {'size': FONT_SIZE_ANIMATION_BUTTON, 'color': PALETTE.get("font")},
                    'buttons': [
                        {'label': "▶", 'method': "animate", 'args': [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300, "easing": "cubic-in-out"}}]},
                        {'label': "❚❚", 'method': "animate", 'args': [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
                    ],
                    'pad': {"r": 10, "l": 5, "t": 0, "b": 0}, 'x': 0.95, 'xanchor': "right", 'y': 0, 'yanchor': "bottom"
                }]
            )
        else: fig.update_layout(sliders=None, updatemenus=None) # Remove controlos se não houver animação
            
        return fig

    # Para Temperatura e Humidade, delega para as funções auxiliares específicas
    if variable == "HUMIDADERELATIVA":
        return _create_humidity_density_map(df_filtered_geo, ano, mes_val, selected_distrito, selected_concelho, PALETTE, MESES_EXTENSO, fig_height=meteo_map_height_numeric)
    if variable == "TEMPERATURA":
        return _create_temperature_density_map(df_filtered_geo, ano, mes_val, selected_distrito, selected_concelho, PALETTE, MESES_EXTENSO, fig_height=meteo_map_height_numeric)

    # Fallback se a variável meteorológica não for suportada
    return create_empty_figure(f"Variável meteorológica '{variable}' não suportada.", height=meteo_map_height_numeric)

# Cria o Modal "Sobre Nós"
def create_about_us_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Sobre Nós")), # Título do modal
        dbc.ModalBody([ 
            html.P("Somos estudantes de Engenharia e Ciência de Dados da Universidade de Coimbra e desenvolvemos o CRONOFOGO, uma marca de visualização interativa sobre os incêndios em Portugal Continental."),
            html.P("Este trabalho é pensado tanto para curiosos, como para o público em geral que queira explorar como têm evoluído os incêndios, ou se haverão incêndios, — e como o tempo influencia (ou poderá influenciar) esses eventos."),
            html.H5("Fonte dos Dados:", className="mt-4"),
            html.P(["Os dados foram obtidos através do ", html.A("Instituto da Conservação da Natureza e das Florestas (ICNF)", href="https://github.com/cityxdev/icnf_fire_data", target="_blank"), " e abrangem o período de 2012 a 2021."]),
            html.H5("Equipa:", className="mt-4"),
            html.Ul([html.Li("Catarina Cruz – catafcruz2002@gmail.com")])
        ]),
        dbc.ModalFooter(dbc.Button("Fechar", id="close-about-us", className="ms-auto", n_clicks=0)) # Botão para fechar
    ], id="modal-about-us", size="lg", scrollable=True, centered=True) # Configurações do modal

# Cria a Sidebar (barra lateral com filtros e controlos)
def create_sidebar() -> dbc.Card:
    default_slider_year = 2021
    if not ANOS: # Caso ANOS esteja vazio (não deveria acontecer com os CSVs fornecidos)
        min_year, max_year = 2012, 2022 # Fallback para min/max do slider
    else:
        min_year, max_year = min(ANOS), max(ANOS)
        if default_slider_year not in ANOS: # Se 2021 não estiver nos dados, escolhe um valor válido como default
            # Se 2022 (ano de previsão) está presente, e há pelo menos mais um ano, usa o penúltimo (que seria 2021).
            # Caso contrário, usa o último ano disponível.
            default_slider_year = ANOS[-2] if len(ANOS) >= 2 and 2022 in ANOS else (max_year if ANOS else 2021)

    initial_marks = {
        a: {"label": str(a), "style": {"color": PALETTE["sidebar_text"], "fontSize": "10px", "textAlign": "center"}}
        for a in ANOS
    }
    if 2022 in initial_marks: # Estilo inicial para 2022 (ano de previsão), se existir
        initial_marks[2022]["style"]["color"] = PALETTE.get("prediction_year_color", PALETTE.get("brand"))
    if default_slider_year in initial_marks: # Estilo inicial para o ano default (2021), se existir
         initial_marks[default_slider_year]["style"]["fontWeight"] = "bold" # Negrito para o ano selecionado

    # Retorna o Card da Sidebar
    return dbc.Card([
        html.Div(html.Img(src=LOGO_SRC, style={"width": "60%", "margin": "0px auto 0px auto", "display": "block"}), style={"textAlign": "center",   "paddingTop": "0px", "marginTop": "-12px"}), # Logótipo
        html.Div([ # Secção de filtros principais
            html.Label("Analisar por:", style={"color": PALETTE["sidebar_text"], "fontWeight": "bold", "fontSize": "0.8rem", "marginBottom": "4px", "display": "block", "textAlign": "left"}),
            dbc.RadioItems(id="radio-metrica", options=METRICAS_OPCOES, value="NUM_INCENDIOS", labelStyle={"color": PALETTE["sidebar_text"], "fontSize": "0.75rem"}, className="metrica-radio-custom"), # Seletor de Métrica
            html.Hr(style={"backgroundColor": "rgba(255,255,255,0.2)", "height": "1px", "border": "none", "width": "80%", "margin": "10px auto"}), # Divisor
            html.Label("Selecionar período:", style={"color": PALETTE["sidebar_text"], "fontWeight": "bold", "fontSize": "0.8rem", "marginBottom": "4px", "display": "block", "textAlign": "left"}),
            dbc.Row([ # Filtros de Mês e Ano
                dbc.Col(dbc.RadioItems(id="radio-mes", options=MES_RADIO_OPCOES, value=0, inline=False, className="mes-radio-custom"), width=5), # Seletor de Mês
                dbc.Col(dcc.Slider( # Slider de Ano
                            id="slider-ano",
                            min=min_year, max=max_year, step=None, # step=None para apenas usar as marcas
                            value=default_slider_year, # Valor inicial 2021
                            vertical=True, verticalHeight=250, # Slider vertical
                            marks=initial_marks, # : Usa as marcas iniciais definidas acima
                            className="slider-ano-custom"
                        ), width=7),
            ], className="gx-1", align="start"),
            html.Hr(style={"backgroundColor": "rgba(255,255,255,0.2)", "height": "1px", "border": "none", "width": "80%", "margin": "10px auto"}),
        ], style={"paddingLeft": "30px", "paddingRight": "5px"}),
        html.Div([ # Botões de Ajuda e Sobre (no fundo da sidebar)
            dbc.Row([
                dbc.Col(dbc.Button([html.I(className="fas fa-question-circle me-1"), " Ajuda"], id="btn-help", color="light", outline=True, className="w-100", style={"fontSize": "0.7rem", "padding": "4px 6px", "marginTop": "-20px"}), width=6),
                dbc.Col(dbc.Button([html.I(className="fas fa-info-circle me-1"), " Sobre"], id="btn-about-us", color="light", outline=True, className="w-100", style={"fontSize": "0.7rem", "padding": "4px 6px", "marginTop": "-20px"}), width=6)
            ], className="gx-2")
        ], className="mt-auto", style={"padding": "15px"}) # mt-auto para alinhar ao fundo
    ], body=True, style={ # Estilo da Sidebar
        "backgroundColor": PALETTE["brand_dark"], "height": "100vh", "position": "fixed",
        "top": 0, "left": 0, "width": "220px", "zIndex": 1050, # zIndex alto para sobrepor outros elementos
        "boxShadow": "2px 0px 10px rgba(0,0,0,0.1)", "overflowY": "auto", # Sombra e scroll se necessário
        "display": "flex", "flexDirection": "column" # Flexbox para alinhar logótipo em cima e botões em baixo
    })

app = Dash(__name__, title="CRONOFOGO - Incêndios PT", 
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], 
           suppress_callback_exceptions=True) 
server = app.server 

sidebar, main_content, about_us_modal = create_sidebar(), create_main_content(), create_about_us_modal()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False), # Para manipulação de URL (não usado ativamente neste exemplo)
    dcc.Dropdown(id="dd-distrito", options=OPCOES_DISTRITOS, value="Todos", clearable=False, style={"display":"none"}), # Filtro de distrito (global, mas escondido e controlado por cliques no mapa)
    dcc.Store(id='store-selected-concelho', data="Todos"), # Armazena o concelho selecionado globalmente
    dcc.Store(id='store-filtered-data-year-month'), # Armazena os dados filtrados por ano e mês (para otimizar callbacks)
    dcc.Store(id='store-help-mode', data=False), # Armazena o estado do modo de ajuda (ativo/inativo)
    sidebar, 
    main_content, 
    about_us_modal 
])

@callback(Output('store-filtered-data-year-month', 'data'), 
          [Input('slider-ano', 'value'), Input('radio-mes', 'value')])
def update_year_month_store(ano: int, mes_val: int) -> Optional[str]:
    if ano is None or mes_val is None: return None # Se filtros não definidos
    df_f = DF[DF["ANO"] == int(ano)] # Filtra por ano
    if mes_val != 0: # Se um mês específico for selecionado (0 = "Todos os Meses")
        df_f = df_f[df_f["MES"] == int(mes_val)] # Filtra por mês
    return df_f.to_json(orient='split', date_format='iso') # Converte para JSON e armazena

# Callback para atualizar os filtros globais de Distrito e Concelho
# Ativado por cliques no mapa principal ou pelo botão de reset.
@callback([Output("dd-distrito", "value"), Output("store-selected-concelho", "data")],
          [Input("g-mapa", "clickData"), Input("btn-reset-filtros", "n_clicks")],
          [State("dd-distrito", "value"), State("store-selected-concelho", "data"), State("dd-map-granularity", "value")],
          prevent_initial_call=True) # Não executa na carga inicial
def update_global_geo_filters(click_data, reset_clicks, current_dist, current_conc, map_granularity):
    triggered_id = callback_context.triggered_id # Identifica qual Input ativou o callback

    if triggered_id == "btn-reset-filtros": # Se o botão de reset foi clicado
        return "Todos", "Todos" # Reseta distrito e concelho para "Todos"

    if click_data and click_data["points"] and click_data["points"][0].get("customdata"):
        customdata_list = click_data["points"][0]["customdata"] # Obtém customdata do ponto clicado
        if isinstance(customdata_list, (list, tuple)) and len(customdata_list) > 0:
            clicked_name = customdata_list[0] # O nome do local (Distrito ou Concelho) está em customdata[0]
            
            if map_granularity == "DISTRITO": # Se o mapa está a mostrar Distritos
                if isinstance(clicked_name, str) and clicked_name in DISTRITOS:
                    # Se o distrito clicado é diferente do atual, ou se um concelho estava selecionado, atualiza
                    if clicked_name != current_dist or current_conc != "Todos": 
                        return clicked_name, "Todos" # Define distrito e reseta concelho
            elif map_granularity == "CONCELHO": # Se o mapa está a mostrar Concelhos
                if isinstance(clicked_name, str): 
                    # Se o concelho clicado é diferente do atual, atualiza
                    if clicked_name != current_conc: 
                        # Ao clicar num concelho, o filtro de distrito é resetado para "Todos".
                        # Poderia ser mantido se o concelho pertencesse ao distrito já selecionado,
                        # mas isso exigiria lógica adicional para verificar essa pertença.
                        return "Todos", clicked_name # Define concelho e reseta distrito para "Todos"
    
    return no_update, no_update # Se nenhuma condição de atualização for satisfeita

@callback(Output("subtitulo-dinamico", "children"),
          [Input("dd-distrito", "value"), Input("store-selected-concelho", "data"),
           Input("radio-mes", "value"), Input("slider-ano", "value")])
def update_main_subtitle(sel_dist: str, sel_conc: str, mes_val: int, ano_slider_val: int):
    if not all(v is not None for v in [sel_dist, sel_conc, mes_val, ano_slider_val]):
        return "À espera da seleção de filtros..." # Mensagem de fallback

    ano = int(ano_slider_val) # Garante que ano é inteiro

    # Constrói a parte do local da string do subtítulo
    location_html_elements = []
    if sel_conc != "Todos": # Se um concelho está selecionado
        # Tenta encontrar o distrito do concelho para adicionar informação contextual
        dist_of_conc_series = DF[DF['CONCELHO'].str.lower() == sel_conc.lower()]['DISTRITO'].unique()
        dist_name_suffix = f" (Distrito de {dist_of_conc_series[0].title()})" if len(dist_of_conc_series) > 0 else ""
        location_html_elements.extend(["o concelho de ", html.Strong(sel_conc.title() + dist_name_suffix)])
    elif sel_dist != "Todos": # Se um distrito está selecionado
        location_html_elements.extend(["o distrito de ", html.Strong(sel_dist.title())])
    else: # Se "Todos" (Portugal Continental)
        location_html_elements.append(html.Strong("Portugal Continental"))

    # Constrói a parte do período da string do subtítulo
    period_html_elements = []
    if mes_val == 0: # "Todos os Meses"
        period_html_elements.extend(["ano de ", html.Strong(str(ano))])
    else: # Mês específico
        month_name = MESES_EXTENSO.get(int(mes_val), "").capitalize() # Garante que mes_val é int
        period_html_elements.append(html.Strong(f"{month_name} de {ano}"))

    # Verifica se é o ano de previsão (2022)
    is_prediction = (ano == 2022)
    dados_type_str = "previstos" if is_prediction else "" # String "previstos" ou vazia

    # Monta o subtítulo final combinando as partes
    # Exemplo: "Dados de Incêndios previstos para Portugal Continental – ano de 2022"
    final_children = ["Dados de Incêndios ", html.Strong(dados_type_str), " para "]
    final_children.extend(location_html_elements)
    final_children.append(" – ")
    final_children.extend(period_html_elements)

    return html.Span(final_children) # Retorna como um html.Span para permitir formatação (html.Strong)

# Helper: Filtra os dados (do store) com base no distrito/concelho e retorna o DataFrame filtrado e nome do filtro.
def get_chart_df_and_title_name(stored_data_json: str, sel_dist: str, sel_conc: str) -> Tuple[pd.DataFrame, str]:
    if not stored_data_json: return pd.DataFrame(), "Portugal Continental" # Se não há dados no store
    try: 
        df_ym_slice = pd.read_json(StringIO(stored_data_json), orient='split') # Carrega JSON do store
    except ValueError: 
        return pd.DataFrame(), "Portugal Continental (erro dados)" # Em caso de erro ao carregar
    
    if df_ym_slice.empty: 
        return df_ym_slice, "Portugal Continental" # Se DataFrame vazio

    active_filter_name = "Portugal Continental"; df_chart = df_ym_slice.copy()
    
    # Aplica filtro de concelho ou distrito
    if sel_conc != "Todos":
        # Filtra comparando em minúsculas para robustez, mas usa o nome original para o título
        df_chart = df_ym_slice[df_ym_slice["CONCELHO"].str.lower() == sel_conc.lower()]
        active_filter_name = f"concelho de {sel_conc.title()}" # Nome para títulos de gráficos
    elif sel_dist != "Todos":
        df_chart = df_ym_slice[df_ym_slice["DISTRITO"].str.lower() == sel_dist.lower()]
        active_filter_name = f"distrito de {sel_dist.title()}"
        
    return df_chart, active_filter_name

# Callback para atualizar o estilo das marcas (labels) do slider de ano.
# Destaca o ano selecionado (negrito) e o ano de previsão (cor diferente).
@callback(Output("slider-ano", "marks"),
          [Input("slider-ano", "value")]) # Atualiza quando o valor do slider muda
def update_slider_marks_style(selected_year_value: int) -> Dict[int, Dict[str, Any]]:
    marks = {}
    prediction_color = PALETTE.get("prediction_year_color", "#FFA500") # Cor para 2022 (previsão)
    default_text_color = PALETTE["sidebar_text"] # Cor padrão para outros anos (branco)

    default_style_template = {"fontSize": "10px", "textAlign": "center", "fontWeight": "normal"} # Estilo base

    for year_mark in ANOS: # Itera sobre todos os anos disponíveis
        current_style = default_style_template.copy() # Começa com o estilo base

        # 1. Define a cor base da label do ano
        if year_mark == 2022: # Ano de previsão
            current_style["color"] = prediction_color
        else: # Outros anos
            current_style["color"] = default_text_color

        # 2. Aplica negrito se o ano estiver selecionado, mantendo a cor já definida
        if year_mark == selected_year_value:
            current_style["fontWeight"] = "bold"
            # Se o ano selecionado for 2022, ele já tem `prediction_color` e agora também `fontWeight: "bold"`.
            # Se for outro ano selecionado, terá `default_text_color` e `fontWeight: "bold"`.

        marks[year_mark] = {"label": str(year_mark), "style": current_style}
    return marks

# Callback para os botões segmentados do card Pie/Cloud (Tipo de Causa / Família de Causa)
# Atualiza o dcc.Store que controla qual visualização é mostrada e o estado 'active' dos botões.
@callback(
    [Output("radio-pie-cloud-selector", "data", allow_duplicate=True), # `data` do dcc.Store
     Output("btn-pie-tipo", "active"), Output("btn-pie-familia", "active")], # Estado dos botões
    [Input("btn-pie-tipo", "n_clicks"), Input("btn-pie-familia", "n_clicks")],
    prevent_initial_call=True)
def update_pie_cloud_selector_and_buttons(n_tipo, n_familia):
    button_id = callback_context.triggered_id # Qual botão foi clicado
    if button_id == "btn-pie-tipo": return "tipo", True, False # Seleciona "tipo", ativa botão Tipo
    elif button_id == "btn-pie-familia": return "familia", False, True # Seleciona "familia", ativa botão Família
    return no_update, no_update, no_update # Nenhum botão clicado (raro com prevent_initial_call)

# Callback para os botões segmentados do card Mapa Meteorológico (Temperatura / Humidade / Vento)
# Semelhante ao anterior, para selecionar a variável meteorológica.
@callback(
    [Output("rd-meteo-var", "data", allow_duplicate=True),
     Output("btn-meteo-temp", "active"), Output("btn-meteo-hum", "active"), Output("btn-meteo-vento", "active")],
    [Input("btn-meteo-temp", "n_clicks"), Input("btn-meteo-hum", "n_clicks"), Input("btn-meteo-vento", "n_clicks")],
    prevent_initial_call=True)
def update_meteo_var_selector_and_buttons(n_temp, n_hum, n_vento):
    button_id = callback_context.triggered_id
    if button_id == "btn-meteo-temp": return "TEMPERATURA", True, False, False
    elif button_id == "btn-meteo-hum": return "HUMIDADERELATIVA", False, True, False
    elif button_id == "btn-meteo-vento": return "VENTOINTENSIDADE", False, False, True
    return no_update, no_update, no_update, no_update

# Callback para atualizar o label do toggle de nomes no mapa principal
# Mostra "Nomes dos Distritos" ou "Nomes dos Concelhos" conforme a granularidade selecionada.
@callback(Output('map-text-toggle-label', 'children'), 
          [Input('dd-map-granularity', 'value')])
def update_map_text_toggle_label(granularity_value: str) -> str:
    if granularity_value == "DISTRITO": return "Nomes dos Distritos"
    else: return "Nomes dos Concelhos"

# Reseta todos os filtros para os valores padrão, incluindo o ano para 2021.
@callback([Output("radio-mes", "value", allow_duplicate=True),
           Output("slider-ano", "value", allow_duplicate=True),
           Output("dd-distrito", "value", allow_duplicate=True),
           Output("store-selected-concelho", "data", allow_duplicate=True),
           Output("dd-map-granularity", "value", allow_duplicate=True)],
          [Input("btn-reset-filtros", "n_clicks")],
          prevent_initial_call=True)
def reset_all_filters_on_button_click(n_clicks):
    if n_clicks is None or n_clicks == 0: # Se o botão não foi clicado (ou é o clique inicial)
        return no_update, no_update, no_update, no_update, no_update
    
    # Define 2021 como ano padrão no reset.
    # Se 2021 não estiver em ANOS (improvável), usa um fallback.
    default_reset_year = 2021
    if not ANOS: # ANOS vazio
        pass # default_reset_year já é 2021, mas o slider pode não ter este valor
    elif default_reset_year not in ANOS:
         # Se 2021 não está em ANOS, usar o penúltimo se 2022 for o último, senão o último ano disponível.
        default_reset_year = ANOS[-2] if len(ANOS) >= 2 and 2022 in ANOS else (max(ANOS) if ANOS else 2021)

    # Retorna os valores padrão para todos os filtros
    return 0, default_reset_year, "Todos", "Todos", "DISTRITO" # Mês "Todos", Ano 2021, Local "Todos", Granularidade "DISTRITO"

# Callback para abrir/fechar o modal "Sobre Nós"
@callback(Output("modal-about-us", "is_open"),
          [Input("btn-about-us", "n_clicks"), Input("close-about-us", "n_clicks")], # Inputs: botões de abrir e fechar
          [State("modal-about-us", "is_open")], prevent_initial_call=True) # State: estado atual do modal
def toggle_about_us_modal(open_clicks, close_clicks, is_open):
    if callback_context.triggered_id in ["btn-about-us", "close-about-us"]: # Se um dos botões foi clicado
        return not is_open # Inverte o estado do modal (abre se fechado, fecha se aberto)
    return is_open # Caso contrário, mantém o estado atual

# --- Callbacks para Atualizar Gráficos (com suporte ao Modo de Ajuda) ---
# Estes callbacks têm uma estrutura semelhante:
# 1. Verificam se o modo de ajuda está ativo. Se sim, mostram o texto de ajuda.
# 2. Verificam se todos os filtros necessários estão definidos.
# 3. Carregam os dados filtrados do dcc.Store.
# 4. Chamam a função de criação do gráfico correspondente.
# 5. Retornam o gráfico dentro de um dcc.Loading para feedback visual.

# Callback para o Mapa Principal
@callback(
    Output("display-area-mapa", "children"), # Onde o mapa ou texto de ajuda será renderizado
    Output("mapa-dynamic-title", "children"), # Título dinâmico do card do mapa
    [Input("radio-metrica", "value"), Input("slider-ano", "value"), Input("radio-mes", "value"),
     Input("map-text-toggle", "value"), Input('store-filtered-data-year-month', 'data'),
     Input('dd-map-granularity', 'value'), Input('store-help-mode', 'data')], # Inputs de filtros e modo de ajuda
    [State('store-main-map-height', 'data')] # Altura do mapa (do dcc.Store)
)
def update_main_map(metric, ano, mes_val, show_names, stored_data_json, map_granularity, help_mode_active, map_height_px):
    map_h_val = map_height_px if map_height_px else int(MAIN_MAP_FIXED_HEIGHT.replace("px","")) # Obtém altura

    title_metric_val = TITULOS_METRICAS.get(metric, "dados") # Nome da métrica para o título
    gran_text_val = "distrito" if map_granularity == "DISTRITO" else "concelho" if map_granularity == "CONCELHO" else "localização"
    dynamic_title = f"{title_metric_val} por {gran_text_val.lower()}" # Monta o título

    if help_mode_active: # Se modo de ajuda ativo
        return create_help_text_div(HELP_TEXTS["g-mapa"], map_h_val, "g-mapa"), dynamic_title # Mostra texto de ajuda

    # Se modo de ajuda inativo, gera o gráfico
    if map_granularity is None: fig = create_empty_figure("Por favor, selecione o nível do mapa<br>(distrito/concelho).", height=map_h_val)
    elif not all([metric, ano is not None, mes_val is not None, show_names is not None]): fig = create_empty_figure("Aguardando seleção de filtros...", height=map_h_val)
    elif not stored_data_json: fig = create_empty_figure("Aguardando dados...", height=map_h_val)
    else: # Se tudo OK, gera o mapa
        try: df_filtered = pd.read_json(StringIO(stored_data_json), orient='split') # Carrega dados
        except ValueError: fig = create_empty_figure("Erro ao carregar dados.", height=map_h_val)
        else: fig = fig_mapa(df_filtered, metric, int(ano), int(mes_val), show_names, map_granularity) # Chama função do mapa

    return dcc.Loading(dcc.Graph(id="g-mapa", figure=fig, config={'displayModeBar': False}, style={"height": f"{map_h_val}px"})), dynamic_title

# Callback para o Gráfico de Perfil Horário
@callback(Output("display-area-perfil-horario", "children"),
    [Input("radio-metrica", "value"), Input("dd-distrito", "value"), Input("store-selected-concelho", "data"),
     Input("slider-ano", "value"), Input("radio-mes", "value"), Input('store-filtered-data-year-month', 'data'),
     Input('store-help-mode', 'data')],
    [State('store-perfil-horario-height', 'data')]
)
def update_profile_chart(metric, sel_dist, sel_conc, ano, mes_val, stored_data_json, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else 200 # Altura do gráfico

    if help_mode_active: # Modo de ajuda
        return create_help_text_div(HELP_TEXTS["g-perfil-horario"], chart_h_val, "g-perfil-horario")

    # Gera o gráfico
    if not all(v is not None for v in [metric, sel_dist, sel_conc, ano, mes_val]): fig = create_empty_figure("Filtros incompletos.", height=chart_h_val)
    elif not stored_data_json: fig = create_empty_figure("Aguardando dados...", height=chart_h_val)
    else:
        df_chart, title_name = get_chart_df_and_title_name(stored_data_json, sel_dist, sel_conc) # Filtra dados por local
        if df_chart.empty and stored_data_json: 
            time_period = get_time_period_string(int(ano), int(mes_val))
            fig = create_empty_figure(f"Sem dados para perfil horário<br>({title_name.lower()} - {time_period.lower()}).", height=chart_h_val)
        else: fig = fig_perfil_horario(df_chart, metric, title_name, int(ano), int(mes_val), height=chart_h_val) # Gera gráfico
    
    return dcc.Loading(dcc.Graph(id="g-perfil-horario", figure=fig, config={'displayModeBar': False}, style={"height": f"{chart_h_val}px"}))

# Callback para o Mapa Meteorológico (pequeno) e seu título
@callback(
    Output("display-area-meteo-map", "children"), Output("meteo-map-title-dynamic", "children"),
    [Input("rd-meteo-var", "data"), Input("slider-ano", "value"), Input("radio-mes", "value"),
     Input("dd-distrito", "value"), Input("store-selected-concelho", "data"), Input('store-help-mode', 'data')],
    [State('store-meteo-map-height', 'data')]
)
def update_small_meteo_map_and_title(variable, ano, mes_val, sel_dist, sel_conc, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else int(METEO_MAP_NEW_HEIGHT.replace("px","")) # Altura
    
    # Define o título dinâmico com base na variável meteorológica selecionada
    default_title = "Mapa Meteorológico"
    if variable == "TEMPERATURA": title = "Distribuição da Temperatura Diária"
    elif variable == "HUMIDADERELATIVA": title = "Distribuição da Humidade Rel. Diária"
    elif variable == "VENTOINTENSIDADE": title = "Intensidade e Direção do Vento Diária"
    else: title = default_title

    if help_mode_active: # Modo de ajuda
        return create_help_text_div(HELP_TEXTS["g-meteo-map"], chart_h_val, "g-meteo-map"), title

    # Gera o mapa
    if not all(v is not None for v in [variable, ano, mes_val, sel_dist, sel_conc]): fig = create_empty_figure("Filtros incompletos.", height=chart_h_val)
    elif int(mes_val) == 0: # Mapas meteo requerem um mês específico
        fig = create_empty_figure("⚠️<br>Seleciona um mês específico<br>para ver o mapa meteorológico.", height=chart_h_val)
    else: 
        # Usa o DataFrame global (DF) para os mapas meteo, pois eles podem mostrar dados de dias específicos
        # que podem não estar no `store-filtered-data-year-month` se este agregar por mês.
        fig = fig_meteo_map(DF, variable, sel_dist, sel_conc, int(ano), int(mes_val)) 
    
    return dcc.Loading(dcc.Graph(id="g-meteo-map", figure=fig, config={'displayModeBar': False}, style={"height": f"{chart_h_val}px"})), title

# Callback para o Gráfico de Dispersão (Temperatura vs Humidade por Vento)
@callback(Output("display-area-scatter-meteo", "children"),
    [Input("slider-ano", "value"), Input("dd-distrito", "value"), Input("store-selected-concelho", "data"),
     Input("radio-mes", "value"), Input('store-filtered-data-year-month', 'data'),
     Input("rangeslider-scatter-wind-filter", "value"), Input('store-help-mode', 'data')], # Input do RangeSlider de Vento
    [State('store-scatter-meteo-height', 'data')]
)
def update_scatter_meteo_chart(ano, sel_dist, sel_conc, mes_val, stored_data_json, selected_wind_range, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else 305 # Altura

    if help_mode_active: # Modo de ajuda
        return create_help_text_div(HELP_TEXTS["g-scatter-meteo"], chart_h_val, "g-scatter-meteo")

    # Gera o gráfico
    if not all(v is not None for v in [ano, sel_dist, sel_conc, mes_val, selected_wind_range]): fig = create_empty_figure("Filtros incompletos.", height=chart_h_val)
    elif not stored_data_json: fig = create_empty_figure("Aguardando dados...", height=chart_h_val)
    else:
        df_chart, title_name = get_chart_df_and_title_name(stored_data_json, sel_dist, sel_conc)
        if df_chart.empty: # Se df_chart estiver vazio após get_chart_df_and_title_name
            time_period = get_time_period_string(int(ano), int(mes_val))
            fig = create_empty_figure(f"Sem dados para Temp/Hum/Vento<br>({title_name.lower()} - {time_period.lower()}).", height=chart_h_val)
        else:
            fig = fig_scatter_meteo(df_chart, title_name, int(ano), int(mes_val), height=chart_h_val) # Gera gráfico base
            
            # Filtra os pontos no gráfico com base no RangeSlider de Vento (ajustando opacidade)
            if fig.data: # Verifica se o gráfico tem dados (traces)
                opacity_visible = 0.8; opacity_hidden = 0.05 # Opacidades para pontos dentro/fora do intervalo
                min_wind, max_wind = selected_wind_range # Intervalo de vento do slider
                
                for trace_idx, trace in enumerate(fig.data): # Itera sobre os traces (Agrícola, Florestal)
                    if hasattr(trace, 'customdata') and trace.customdata is not None and len(trace.customdata) > 0:
                        # Extrai os valores de vento do customdata de cada ponto no trace
                        wind_values_for_trace = []
                        for pcd_item in trace.customdata: # customdata é geralmente uma lista de listas/arrays
                            if isinstance(pcd_item, (list, tuple, np.ndarray)) and len(pcd_item) > 0:
                                wind_values_for_trace.append(pcd_item[0]) # Vento está em customdata[0]
                            elif pd.notna(pcd_item): # Fallback se for escalar
                                wind_values_for_trace.append(pcd_item)
                            else: wind_values_for_trace.append(np.nan) # Adiciona NaN se não conseguir extrair

                        # Define opacidade para cada ponto com base no intervalo de vento
                        opacities = [opacity_visible if pd.notna(val) and min_wind <= val <= max_wind else opacity_hidden for val in wind_values_for_trace]
                        
                        if opacities: # Se houver opacidades calculadas
                            if hasattr(trace.marker, 'opacity'): fig.data[trace_idx].marker.opacity = opacities # Define opacidade por ponto
                            else: fig.data[trace_idx].marker = {'opacity': opacities} # Fallback
                        else: # Se não foi possível calcular opacidades
                            if hasattr(trace.marker, 'opacity'): fig.data[trace_idx].marker.opacity = opacity_hidden
                            else: fig.data[trace_idx].marker = {'opacity': opacity_hidden}
                    else: # Se não houver customdata no trace
                        if hasattr(trace.marker, 'opacity'): fig.data[trace_idx].marker.opacity = opacity_hidden
                        else: fig.data[trace_idx].marker = {'opacity': opacity_hidden}
    
    return dcc.Loading(dcc.Graph(id="g-scatter-meteo", figure=fig, config={'displayModeBar': False}, style={"height": f"{chart_h_val}px"}))


# Callback para o Gráfico de Violino
@callback(Output("display-area-violin", "children"),
    [Input("slider-ano", "value"), Input("dd-distrito", "value"), Input("store-selected-concelho", "data"),
     Input("radio-mes", "value"), Input('store-filtered-data-year-month', 'data'), Input('store-help-mode', 'data')],
    [State('store-violin-height', 'data')]
)
def update_violin_chart(ano, sel_dist, sel_conc, mes_val, stored_data_json, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else 305 # Altura

    if help_mode_active: # Modo de ajuda
        return create_help_text_div(HELP_TEXTS["g-violin"], chart_h_val, "g-violin")

    if not all(v is not None for v in [ano, sel_dist, sel_conc, mes_val]): fig = create_empty_figure("Filtros incompletos.", height=chart_h_val)
    elif not stored_data_json: 
        fig = create_empty_figure("Aguardando dados...", height=chart_h_val)
    else:
        df_chart, title_name = get_chart_df_and_title_name(stored_data_json, sel_dist, sel_conc)
        if df_chart.empty and stored_data_json : # Se df_chart vazio mas havia dados no store
            time_period = get_time_period_string(int(ano), int(mes_val))
            fig = create_empty_figure(f"Sem dados meteorológicos para<br>distribuição ({title_name.lower()} - {time_period.lower()}).", height=chart_h_val)
        else: fig = fig_violin_distribution(df_chart, title_name, int(ano), int(mes_val), height=chart_h_val) # Gera gráfico
    
    return dcc.Loading(dcc.Graph(id="g-violin", figure=fig, config={'displayModeBar': False}, style={"height": f"{chart_h_val}px"}))

# Callback para o Card de Causas (alterna entre Gráfico de Pizza e Nuvem de Palavras)
@callback(
    Output("pie-cloud-display-area", "children"), Output("pie-cloud-title", "children"), # Conteúdo e título do card
    [Input("radio-pie-cloud-selector", "data"), Input("radio-metrica", "value"), # Seleção Tipo/Família e Métrica
     Input("slider-ano", "value"), Input("dd-distrito", "value"),
     Input("store-selected-concelho", "data"), Input("radio-mes", "value"),
     Input('store-filtered-data-year-month', 'data'), Input('store-help-mode', 'data')],
    [State("store-pie-cloud-height", "data")]
)
def toggle_pie_cloud(selected_view_type, metric_val, ano, sel_dist, sel_conc, mes_val, stored_data_json, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else 200 # Altura
    show_word_cloud = (selected_view_type == "familia") # True se "Família" selecionado, False se "Tipo"

    # Define o título dinâmico do card
    metric_suffix_map = {"NUM_INCENDIOS": "por Nº de Incêndios", "AREA_ARDIDA": "por Área Ardida", "DURACAO_MEDIA": "por Duração Total"}
    title_suffix = metric_suffix_map.get(metric_val, "por Nº de Incêndios")
    title_text_base = "Famílias de Causa" if show_word_cloud else "Tipos de Causa"
    title_text = f"{title_text_base} {title_suffix}"

    if help_mode_active: # Modo de ajuda
        help_key = "pie-cloud-familia" if show_word_cloud else "pie-cloud-tipo" # Chave para o texto de ajuda
        return create_help_text_div(HELP_TEXTS[help_key], chart_h_val, help_key), title_text

    # Estilo para mensagens de erro/aviso
    message_div_style = {"textAlign": "center", "padding": "20px", "height": f"{chart_h_val}px", "display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center", "color": PALETTE["font"], "fontSize": "13px", "lineHeight": "1.6"}
    
    # Validação de inputs
    if not all(v is not None for v in [selected_view_type, metric_val, ano, sel_dist, sel_conc, mes_val]): 
        return html.Div(dcc.Markdown("Filtros incompletos."), style=message_div_style), title_text
    if not stored_data_json: 
        return html.Div(dcc.Markdown("Aguardando dados..."), style=message_div_style), title_text

    df_chart, title_name_for_data_context = get_chart_df_and_title_name(stored_data_json, sel_dist, sel_conc) # Filtra dados
    
    if df_chart.empty:
        time_period = get_time_period_string(int(ano), int(mes_val))
        empty_msg_content = f"Sem dados de causas para {title_name_for_data_context.lower()}<br>({time_period.lower()})."
        return html.Div(dcc.Markdown(empty_msg_content.replace("<br>", "\n")), style=message_div_style), title_text

    # Gera a visualização apropriada (Nuvem ou Pizza)
    if show_word_cloud: # Nuvem de Palavras
        img_src_or_msg = img_nuvem_palavras(df_chart, metric_val, title_name_for_data_context, int(ano), int(mes_val), width=400, height=chart_h_val)
        if isinstance(img_src_or_msg, str) and img_src_or_msg.startswith("data:image/png;base64,"): # Se for uma imagem base64
            return html.Img(src=img_src_or_msg, style={"width":"100%","height":f"{chart_h_val}px","objectFit":"contain"}), title_text
        else: # Se for uma mensagem de erro/aviso da função da nuvem
            return html.Div(dcc.Markdown(img_src_or_msg.replace("<br>", "\n").replace("\n", "<br>")), style=message_div_style), title_text
    else: # Gráfico de Pizza
        fig = fig_pie_causas(df_chart, metric_val, title_name_for_data_context, int(ano), int(mes_val), height=chart_h_val)
        return dcc.Loading(dcc.Graph(id="g-pie-causas", figure=fig, config={'displayModeBar':False}, style={"height":f"{chart_h_val}px"})), title_text

# Callback para o Gráfico de Relação entre Métricas (barras + linha)
@callback(Output("display-area-relacao-metricas", "children"),
    [Input("slider-ano", "value"), Input("dd-distrito", "value"), # Filtros de ano, distrito
     Input("store-selected-concelho", "data"), Input('store-help-mode', 'data')], # e concelho (do store)
    [State('store-relacao-metricas-height', 'data')]
)
def update_relacao_metricas_chart(ano_slider_val, sel_dist, sel_conc, help_mode_active, chart_height_px):
    chart_h_val = chart_height_px if chart_height_px else 250 # Altura

    if help_mode_active: # Modo de ajuda
        return create_help_text_div(HELP_TEXTS["g-relacao-metricas"], chart_h_val, "g-relacao-metricas")

    # Gera o gráfico
    if not all(v is not None for v in [ano_slider_val, sel_dist, sel_conc]): fig = create_empty_figure("Aguardando seleção de filtros.", height=chart_h_val)
    else:
        ano = int(ano_slider_val)
        df_ano_filtrado = DF[DF["ANO"] == ano].copy() # Filtra o DataFrame GLOBAL pelo ano (este gráfico mostra dados de todos os meses)
        
        if df_ano_filtrado.empty: fig = create_empty_figure(f"Sem dados para o ano de {ano}.", height=chart_h_val)
        else:
            active_filter_name_for_title = "Portugal Continental"
            df_chart_data = df_ano_filtrado # Começa com todos os dados do ano
            
            if sel_conc != "Todos": # Se um concelho está selecionado
                df_chart_data = df_ano_filtrado[df_ano_filtrado["CONCELHO"].str.lower() == sel_conc.lower()]
                # Adiciona nome do distrito ao título do concelho para contexto
                dist_of_conc = DF[DF['CONCELHO'].str.lower() == sel_conc.lower()]['DISTRITO'].unique()
                dist_suffix = f" (Dist. {dist_of_conc[0].title()})" if len(dist_of_conc) > 0 else ""
                active_filter_name_for_title = f"Concelho de {sel_conc.title()}{dist_suffix}"
            elif sel_dist != "Todos": # Se um distrito está selecionado
                df_chart_data = df_ano_filtrado[df_ano_filtrado["DISTRITO"].str.lower() == sel_dist.lower()]
                active_filter_name_for_title = f"Distrito de {sel_dist.title()}"
            
            fig = fig_relacao_metricas(df_chart_data, ano, active_filter_name_for_title, altura_grafico=chart_h_val) # Gera gráfico
            
    return dcc.Loading(dcc.Graph(id="g-relacao-metricas", figure=fig, config={'displayModeBar': False}, style={"height": f"{chart_h_val}px"}))

# --- Callback do Botão de Ajuda ---
# Ativa/desativa o modo de ajuda e altera o texto/estilo do botão.
@callback(
    [Output('store-help-mode', 'data'), Output('btn-help', 'children'), # Estado do modo de ajuda e aparência do botão
     Output('btn-help', 'color'), Output('btn-help', 'outline')],
    [Input('btn-help', 'n_clicks')], # Ativado pelo clique no botão
    [State('store-help-mode', 'data')], # Obtém o estado atual do modo de ajuda
    prevent_initial_call=True
)
def toggle_help_mode(n_clicks, current_help_mode):
    if n_clicks is None or n_clicks == 0: return no_update, no_update, no_update, no_update # Se não clicado
    
    new_help_mode = not current_help_mode # Inverte o modo de ajuda
    
    if new_help_mode:
        button_text = [html.I(className="fas fa-times-circle me-1"), " Fechar Ajuda"] 
        button_color = "warning"; button_outline = False 
    else:
        button_text = [html.I(className="fas fa-question-circle me-1"), " Ajuda"]
        button_color = "light"; button_outline = True # Botão normal
        
    return new_help_mode, button_text, button_color, button_outline

if __name__ == '__main__':
    app.run(debug=True) 