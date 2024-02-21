import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import geopandas as gpd

# Reading the data from
netflix_data = pd.read_csv("netflix_titles.csv")

# Preprocessing the data to handle missing values
netflix_data['country'] = netflix_data['country'].fillna(
    netflix_data['country'].mode()[0])
netflix_data['country'] = netflix_data['country'].apply(
    lambda x: x.split(",")[0])
netflix_data['cast'].replace(np.nan, 'No Data', inplace=True)
netflix_data['director'].replace(np.nan, 'No Data', inplace=True)
netflix_data.dropna(inplace=True)

# Changing the rating values to viewer friendly format
netflix_data['rating'] = netflix_data['rating'].replace({
    'PG-13': 'Teens',
    'TV-MA': 'Adults',
    'PG': 'Older Kids',
    'TV-14': 'Teens',
    'TV-PG': 'Older Kids',
    'TV-Y': 'Kids',
    'TV-Y7': 'Older Kids',
    'R': 'Adults',
    'TV-G': 'Kids',
    'G': 'Kids',
    'NC-17': 'Adults',
    'NR': 'Adults',
    'UR': 'Adults'

})

# Dropping Duplicates
netflix_data.drop_duplicates(inplace=True)

# Creating World Map with GeoPandas
df_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

netflix_df_visual = netflix_data[['title', 'country']]

netflix_df_visual = netflix_df_visual.groupby(['country'])["title"]\
    .count().reset_index()\
    .sort_values('title', ascending=False).head(10)


selected_countries = ['United States of America', 'India', 'United Kingdom',
                      'Canada', 'Japan', 'France', 'South Korea', 'Spain',
                      'Mexico', 'Australia']
df_netflix_world = df_world.merge(netflix_df_visual, how="left", left_on=[
    'name'], right_on=['country'])
finaldf = df_netflix_world.sort_values(by='title', ascending=False)
finaldf.head(10)

# Creating the layout using GridSpec
fig = plt.figure(figsize=(10, 18), facecolor='black', constrained_layout=True)
gs = fig.add_gridspec(3, 3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.patch.set_facecolor('k')
ax4 = fig.add_subplot(gs[-1, :-1])

gs2 = gs[1:, -1].subgridspec(3, 1)
ax3_1 = fig.add_subplot(gs2[0])
ax3_2 = fig.add_subplot(gs2[1])
ax3_3 = fig.add_subplot(gs2[2])

# Visualising the Top Countries with highest Netflix usage
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2%", pad="0.5%")
finaldf.plot(column='title', ax=ax1, cax=cax,
             legend=True,
             missing_kwds={"color": "lightgrey", "edgecolor": "white"})
ax1.set_title("NETFLIX DATA ANALYSIS", color='red', fontsize=18)
ax1.set_axis_off()

# Filtering the data based on the release year
netflix_year_filtered = netflix_data.query("release_year >= 2009")
netflix_year_filtered = netflix_year_filtered.groupby(
    "release_year")["show_id"].count().reset_index()

# Visualising the Content Release trend over the years
sns.barplot(data=netflix_year_filtered, x='release_year', y='show_id',
            ax=ax4, color='#e50914',)
ax4.tick_params(axis='x', colors='white')
ax4.tick_params(axis='y', colors='white')
ax4.set_facecolor("black")
ax4.set_ylabel('Number of Netflix Content', color='white')
ax4.set_xlabel('Release Year', color='white')

# Retrieving the number of Netflix content for USA, UK and India
data_usa = netflix_data.query('country == "United States of America"')
usa_count = data_usa.groupby("type")['show_id'].count().reset_index()
data_india = netflix_data.query('country == "India"')
india_count = data_india.groupby("type")['show_id'].count().reset_index()
data_uk = netflix_data.query('country == "United Kingdom"')
uk_count = data_uk.groupby("type")['show_id'].count().reset_index()

# Visualising the split between the Movies/TV Shows for USA, UK and India
explode = (0.07, 0.07)

color_palette = ['#b20710', '#f5f5f1']
ax3_1.pie(usa_count['show_id'], autopct='%1.1f%%', startangle=90,
          pctdistance=0.5, explode=explode, colors=color_palette,
          labels=usa_count['type'], wedgeprops=dict(width=0.25, edgecolor='w'))
ax3_1.set_title('United States of America', color='red')
ax3_1.add_artist(plt.Circle((0, 0), 0.1, fc='black'))

ax3_2.pie(india_count['show_id'], autopct='%1.1f%%', startangle=90,
          pctdistance=0.5, explode=explode, colors=color_palette,
          labels=india_count['type'], wedgeprops=dict(width=0.25, edgecolor='w'))
ax3_2.set_title('India', color='red')

ax3_2.add_artist(plt.Circle((0, 0), 0.1, fc='black'))
ax3_3.patch.set_facecolor('black')

# Visualising the variation in number of contents for
# different categories in selected countries
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ['#221f1f', '#b20710'])
df_heatmap = pd.crosstab(
    netflix_data['country'], netflix_data['rating'], normalize="index").T
selected_rating = ['Kids', 'Older Kids', 'Teens', 'Adults']
heat_map = sns.heatmap(df_heatmap.loc[selected_rating, selected_countries],
                       cmap=cmap, square=True, linewidth=2.5, cbar=False,
                       annot=True, fmt='1.0%', vmax=.6, vmin=0.05, ax=ax2,
                       annot_kws={"fontsize": 14})
heat_map.set_xlabel('')
heat_map.set_ylabel('')
heat_map.set_xticklabels(heat_map.get_xticklabels(), color='white')
heat_map.set_yticklabels(heat_map.get_yticklabels(), color='white')

fig.text(0.07, 0.64, 'Proportion of Content for Age groups based on Country',
         fontsize=12, fontweight='bold', fontfamily='serif')
fig.text(0.07, 0.59,
         '''We can see that Kids content has the lowest proportion for 
all the countries whereas Adult content is predominant in most 
of the countries except Japan and India.
''', fontsize=10, fontweight='light', fontfamily='serif', color='white')


fig.text(0.09, 0.34, 'Movies & TV Shows added over time', fontsize=12,
         fontweight='bold', fontfamily='serif')
fig.text(0.09, 0.24,
         '''
We see a slow start for Netflix 
over several years. Things begin 
to pick up in 2015 and then there 
is a rapid increase from 2016.

It looks like content additions 
have slowed down in 2020, 
likely due to the COVID-19 pandemic.

''', fontsize=10, fontweight='light', fontfamily='serif', color='white')

fig.text(0.72, 0.43,
         '''
In both the countries, 
Movies are the predominent content,
but India has very little content 
when it comes to 
TV Shows compared to USA 

''', fontsize=10, fontweight='light', fontfamily='serif', color='white')

fig.text(0.72, 0.2, 'CONCLUSION', fontsize=12,
         fontweight='bold', fontfamily='serif', color='red')
fig.text(0.72, 0.05,
         '''
Over the years Netflix has gained popularity
as one of the leading OTT platform in the world.

Few countries like USA, India and UK
are the primary audience for Netflix.

Netflix saw a dip in content generation during 
the COVID pandemic.

Adults are the primary target audience for 
the platform.

Netflix favours Movies over TV Show in most of the
countries which could be due to higher Revenue gained 
from Movies.
''', fontsize=10, fontweight='light', fontfamily='serif', color='white')

fig.text(0.40, 0.72, 'Popularity of Netflix in the World', fontsize=12,
         fontweight='bold', fontfamily='serif')

fig.text(0.07, 0.65,
         '''Netflix can be described as the most popular OTT platform currently available
in the market.
Netflix is gaining popularity among the different nations which is depicted
in the graph above.
''', fontsize=10, fontweight='light', fontfamily='serif', color='white')

fig.text(0.85, 0.04, 'Jeffy Jaxon (21082185)', fontsize=12,
         fontweight='bold', fontfamily='serif', color='white')

fig.set_facecolor('k')
plt.savefig("21082185.png", dpi=300, bbox_inches='tight')
plt.show()
