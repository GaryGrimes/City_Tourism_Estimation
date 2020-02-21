import plotly.graph_objects as go
import urllib, json

url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
fig = go.Figure(data=[go.Sankey(
    valueformat=".0f",
    valuesuffix="TWh",
    # Define nodes
    node=dict(
        pad=15,
        thickness=15,
        line=dict(color="black", width=0.5),
        label=data['data'][0]['node']['label'],
        color=data['data'][0]['node']['color']
    ),
    # Add links
    link=dict(
        source=data['data'][0]['link']['source'],
        target=data['data'][0]['link']['target'],
        value=data['data'][0]['link']['value'],
        label=data['data'][0]['link']['label']
    ))])

fig.update_layout(
    title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
    font_size=10)
fig.show()
