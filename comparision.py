# -*- coding: utf-8 -*-
"""
Created on Sat May  4 12:37:24 2019

@author: sudhanshu kumar sinh
"""
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import output_file,show
import numpy as np

data =pd.read_csv("wine.csv")
properties=data.iloc[:100,0:12]

wine_corr=pd.DataFrame.corr(properties.transpose())

from sklearn.cluster.bicluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters=6,random_state=0)
model.fit(wine_corr)

properties["Group"]=pd.DataFrame(model.row_labels_,index=properties.index)
properties=properties.ix[np.argsort(model.row_labels_)]
properties=properties.reset_index(drop=True)

correlations = pd.DataFrame.corr(properties.iloc[:100,0:12].transpose())
correlations = np.array(correlations)

#Need properties and correlations for furter work

cluster_colors = ["red", "orange", "green", "blue", "purple", "gray"]

temp=list(properties.index)
wine_type=list(map(str,temp))

correlation_colors = []
for i in range(len(wine_type)):
    for j in range(len(wine_type)):
        if correlations[i][j] < 0.7 :                
            correlation_colors.append('white')         
        else:                                          
            if properties.Group[i]==properties.Group[j]:                
                correlation_colors.append(cluster_colors[properties.Group[i]]) 
            else:                                      
                correlation_colors.append('lightgray')
                

source = ColumnDataSource(
    data = {
        "x": np.repeat(wine_type,len(wine_type)),
        "y": list(wine_type)*len(wine_type),
        "colors": correlation_colors,
        "correlations": correlations.flatten(),
    }
)

output_file("Wine Correlations.html", title="Wine Correlations")
fig = figure(title="Wine Correlations",
    x_axis_location="above", tools="reset,hover,save",
    x_range=list(reversed(wine_type)), y_range=wine_type)
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.xaxis.major_label_orientation = np.pi / 3

fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='correlations')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Wine": "@x, @y",
    "Correlation": "@correlations",
}
show(fig)