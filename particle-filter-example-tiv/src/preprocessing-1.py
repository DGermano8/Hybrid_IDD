import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as matplotlib
matplotlib.use('QtAgg')
import pandas as pd
import plotnine as p9
from plotnine import *

tcid_df = pd.read_csv("data/tcid.csv")
tcid_df["log10_tcid"] = (tcid_df["log10_tcid"]
                         .astype(str)
                         .apply(lambda x: float(x[2:]) if x.startswith("<=") else float(x)))
tcid_df["is_truncated"] = tcid_df["log10_tcid"] <= 0.5


data_p9 = (ggplot()
           + geom_point(
               data = tcid_df,
               mapping = aes(x = "day",
                             y = "log10_tcid",
                             colour = "is_truncated")
           )
           + geom_hline(yintercept = 0.5, linetype = "dashed")
           + facet_wrap("patient")
           + scale_y_continuous(
               limits = [-4, 8],
               breaks = [2 * i for i in range(-2, 5)]
           )
           + labs(x = "Days post infection",
                  y = "Viral titre (log10 TCID50/ml)")
           + theme_bw()
           + theme(legend_position = "none"))

data_p9.save("out/data-plot.png",
      height = 5.8, width = 8.3 # A5
      # height = 4.1, width = 5.8 # A6
      # height = 2.9, width = 4.1 # A7
             )
