
import plotly.graph_objects as go
import plotly.express as px

def plot_estimated_policy_value(V_clinician_lst, V_WIS_lst, V_WIS_zero_drug_lst, V_WIS_random_lst):
    fig = go.Figure()
    fig.add_trace(go.Box(y=V_clinician_lst, marker_color = 'lightseagreen', name="Clinicians"))
    fig.add_trace(go.Box(y=V_WIS_lst, marker_color='MediumPurple', name="AI"))
    fig.add_trace(go.Box(y=V_WIS_zero_drug_lst, marker_color=px.colors.qualitative.Plotly[5], name="Zero-Drug"))
    fig.add_trace(go.Box(y=V_WIS_random_lst, marker_color=px.colors.qualitative.Plotly[9], name="Random"))

    fig.update_traces(boxpoints='all', # include actual data points
                      pointpos=0,      # add some jitter for a better separation between points
                      jitter=0.1       # relative position of points w.r.t box
                      )
    # add red dot represent the best ai policy value
    fig.add_trace(go.Scatter(x=["AI"], y=[max(V_WIS_lst)], mode='markers', marker=dict(color='red', size=8), name='Best AI'))

    fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="Estimated Policy Value",
            legend_title_text='Policy',
            width=900,
            height=600)

    fig.show()





