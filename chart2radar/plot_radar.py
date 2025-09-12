
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

layout = go.Layout(
    polar={
        # "bgcolor": "rgba(0,0,0,0)",        # set background color
        # "gridshape": "linear",             # set the grid style of the radar
        "radialaxis": {"showticklabels": True, "gridcolor": "grey"},   # grid color
        "angularaxis": {
            # "linecolor": "black",
            "linewidth": 3,
            # "gridcolor": "black",
        },
    },
    # plot_bgcolor="rgba(0,0,0,0)",
)
fig = go.Figure(layout=layout)

# fig = go.Figure()

linecolors = [
    "rgba(255, 0, 0, 0.9)",
    "rgba(0, 0, 255, 0.9)",
    "rgba(255, 0, 255, 0.9)",
    "rgba(0, 255, 255, 0.9)",
    "rgba(8, 46, 84, 0.9)",
]                   # set line color

fillcolors = [
    "rgba(255, 0, 0, 0.1)",
    "rgba(0, 0, 255, 0.1)",
    "rgba(255, 0, 255, 0.1)",
    "rgba(0, 255, 255, 0.1)",
    "rgba(8, 46, 84, 0.1)",
]                     # set fill color

# theta = ['Tumor DSC', 'Organ DSC', 'Tumor NSD', 'Organ NSD', 'Time', 'GPU', 'Final']
theta = ['Lesion NSD', 'Lesion DSC', 'Organ NSD', 'Organ DSC', 'Final Rank', 'GPU Memory', 'Runtime']
# Team Rank

# teams = {
#     'aladdin5': [1, 3, 1, 6, 2, 1, 1],
#     'citi': [3, 1, 4, 1, 6, 1, 2],
#     'blackbean': [4, 10, 3, 2, 4, 1, 3],
#     'hmi306': [8, 7, 8, 3, 1, 1, 4],
#     'hanglok': [6, 6, 6, 5, 7, 1, 5]
# }

teams = {
    'T1-aladdin5': [1, 1, 6, 3, 1, 1, 2],
    'T2-citi': [4, 3, 1, 1, 2, 1, 6],
    'T3-blackbean': [3, 4, 2, 10, 3, 1, 4],
    'T4-hmi306': [8, 8, 3, 7, 4, 1, 1],
    'T5-hanglok': [6, 6, 5, 6, 5, 1, 7]
}

for teamname, fillcolor, linecolor in zip(teams.keys(), fillcolors, linecolors):
    ranks = teams[teamname]  # generate data of each year for plotting
    ranks = [38-x for x in ranks]
    ranks.append(ranks[0])
    theta.append(theta[0])
    fig.add_trace(
        go.Scatterpolar(
            r=ranks,      # set the data to plot radar chart
            theta=theta,  # set the category name of each
            line=go.scatterpolar.Line(color=linecolor, width=2),  # set the line aesthetics
            marker=go.scatterpolar.Marker(color=linecolor, size=5),
            fill="toself",                  # fill the inner part of the content
            fillcolor=fillcolor,            # set the fill color
            name=teamname,       # set the name of the series of data
        )
    )

fig.update_layout(
    width=1000,
    height=800,
    font=dict(family='times new roman', size=24, color="#000000"),
    polar=dict(radialaxis=dict(visible=True, range=[0, 39])),  # set the visibility of radial axis & the range of polar axis
    showlegend=True,      # set the visibility of legend
    legend=dict(y=0, x=0.9),    # set the position of legend
)

# fig.show()
fig.write_image(
    "fig2b_radar.png",
    scale=3,
    engine="kaleido"
)  # write_image to export high-resolution chart, default export engine: orca

