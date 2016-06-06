# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 13:26:28 2016

@author: ReidW
"""


import sys, os, datetime, re
import matplotlib.pyplot as plt
import numpy as np
import samples as s
reload(s)

#Quick compile of recording filename regex
fn = re.compile('recording-(\d+)-(\d+)*');


#day, time for each region
reg_data_dist = [np.empty((0,2)) for i in range(7)]
for root, dirs, files in os.walk(s.RECORD_DIR):
    for jfile in files:
        if os.path.splitext(jfile)[1] == '.wav':
            jpath = os.path.join(root,jfile);

            res = fn.match(jfile);
            date = datetime.date(int(res.groups()[0][0:4]), int(res.groups()[0][4:6]),
                                      int(res.groups()[0][6:8]));
            weekday = date.weekday(); #Monday = 0; Sunday = 6
            time = datetime.time(int(res.groups()[1][0:2]),int(res.groups()[1][2:4]),
                                      int(res.groups()[1][4:6]));

            #Use containing folder to get region (i.e. label);
            jregion = root.split('\\')[-1]

            region = s.Regions_dict[jregion];

            hour_of_day = time.hour + float(time.minute)/60;
            reg_data_dist[region] = np.append(reg_data_dist[region],np.array([[weekday, hour_of_day]]),axis=0)

symbs = ["ro", "bv", "gs", "c^", "mx", "yh", "k+"]
day_ticks = range(7);
day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hour_ticks = np.arange(8,23,2);
hour_labels = ["%02d:00"%hour_ticks[i] for i in range(len(hour_ticks))]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_data_interval(-0.5,6.5)
ax.xaxis.set_view_interval(-0.5,6.5)
ax.xaxis.set_ticks(day_ticks)
ax.xaxis.set_ticklabels(day_labels)
ax.yaxis.set_ticks(hour_ticks)
ax.yaxis.set_ticklabels(hour_labels)
for i in range(7):
    tod = reg_data_dist[i][:,1]
    ax.plot(reg_data_dist[i][:,0],reg_data_dist[i][:,1],symbs[i],label=s.REGIONS[i])
fig.autofmt_xdate()
ax.axes.invert_yaxis()
ax.axes.set_xlim(left=-0.5,right=6.5)
ax.axes.set_ylim(bottom=22.2,top=7.8)
ax.axes.set_ylabel("Hour of Day");
box = ax.get_position()
ax.set_position([box.x0, box.y0,
                 box.width, box.height * 0.8])
ax.legend(loc='upper center', ncol=4,fancybox=True,shadow=True,bbox_to_anchor=(0.48,1.25))