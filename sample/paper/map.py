#!C:/Users/80521/AppData/Local/Programs/Python/Python38 python
# -*- coding=utf8 -*-

'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-04-29 20:09:36
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-04-29 20:09:36
 * @Description: this file is dedicated to defining the data used map.
***********************************************************************'''
from pyecharts.charts import Map,Geo
from pyecharts import options as opts
import random
from pyecharts.globals import GeoType

class MAP():
    def __init__(self, name) -> None:
        self.name = name

    def createCityMap(name_map, attentionRegion, noAttentionRegion):
        # get the max value in list attentionRegion.
        idx, max_value= max(attentionRegion, key=lambda item: item[1])
        (
            Map()
            .add(
                series_name = "attentionRegion",
                data_pair = attentionRegion,
                maptype = "新疆",
                is_map_symbol_show = True,
                name_map = name_map,
                zoom = 1.2,
                itemstyle_opts = {
                    "normal":{"areColor":"white","borderColor":"white"},
                    "emphasis":{"areaColor":"black"}
                }
            )
            .add(
                series_name = "noAttentionRegion",
                data_pair = noAttentionRegion,
                maptype = "新疆"
            )
            .set_global_opts(
                # title_opts = opts.TitleOpts(title = "种植业绿色生产效率",pos_left = 'center'),
                legend_opts = opts.LegendOpts(pos_left = 'left'),
                # '#90EE90','#7BE970','#0FCB11']
                # '#FFB3B3','#FF8686','#D21449']
                # ['#36b101','#75cd00','#cff200','#ffcf00','#ff6700']
                visualmap_opts = opts.VisualMapOpts(max_ = 1.23,is_piecewise = True,range_color = ['#ff6700','#ffcf00','#cff200','#75cd00','#36b101'], textstyle_opts=opts.TextStyleOpts(font_size=20)),
            )
            .set_series_opts(
                label_opts = opts.LabelOpts(is_show = True, color = "black", font_size=13)
            )
            .render("c:/users/80521/desktop/xinJiangMap.html")
        ) 
