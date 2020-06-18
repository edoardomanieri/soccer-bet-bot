import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from dash_table import DataTable
import glob
import os


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.scripts.config.serve_locally = True


app.layout = html.Div([
    html.Button(
        ['Update'],
        id='btn'
    ),
    DataTable(
        id='table',
        data=[]
    )
])


@app.callback(
    [Output("table", "data"), Output('table', 'columns')],
    [Input("btn", "n_clicks")]
)
def updateTable(n_clicks):
    file_path = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(f"{file_path}/../dash/*")
    li = [pd.read_csv(filename, index_col=0, header=0) for filename in files]
    if len(li) == 0:
        return [], []
    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.sort_values(by=['minute'], ascending=False)
    df = df.groupby(['bet_type', 'home']).first().reset_index()
    df = df.sort_values(by=['probability_final'], ascending=False)
    df.drop(columns=['bet_type'], inplace=True)
    columns = [{"name": i, "id": i} for i in df.columns]
    data_ob = df.to_dict('rows')
    return data_ob, columns


if __name__ == '__main__':
    app.run_server(debug=True)
