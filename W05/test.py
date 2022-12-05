#| label: GQ3 TABLE1
#| code-summary: Read and format data
# Include and execute your code here

dat_weather = flights2

#  replace the zeros in the delays late aircraft column with the mean of the column
dat_weather["num_of_delays_late_aircraft"].replace(0, dat_weather["num_of_delays_late_aircraft"].replace(0, np.nan).mean(skipna=True), inplace=True)

# drop rows that have "n/a" for the month
dat_weather = dat_weather[dat_weather.month != 'n/a']
dat_weather.month.dropna(inplace=True)

#  create a new column for a nummeric month value
dat_weather['month_num'] = dat_weather['month']

#  create a new column for a nummeric month value
# replace the month names in month_num with a numeric value
dat_weather['month_num'] = dat_weather['month'].map({'January':1, 'Febuary':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12})

#  create a new column for a weather delays total
dat_weather['delays_weather_sum'] = 0
# add the weather delays to the new column if the month is between April and August, or the month = 0
dat_weather1 = dat_weather.query("month == 0 or (month_num > 3 and month_num < 9)").eval('delays_weather_sum = delays_weather_sum + num_of_delays_weather + (num_of_delays_late_aircraft * .3) + (num_of_delays_nas * .4)')

# add the weather delays to the new column if the month is not between April and August
dat_weather.query('month_num < 4 or month_num > 8').eval('delays_weather_sum = delays_weather_sum + num_of_delays_weather + (num_of_delays_late_aircraft * .3) + (num_of_delays_nas * .65)')

# combine the two dataframes
dat_weather = pd.concat([dat_weather1, dat_weather2])
# create  anew df for a later question
dat_hubs_weather = dat_weather
# group the data by month and sum
dat_weather = dat_weather.groupby("month").sum().reset_index()

dat_weather[['month', 'month_num', 'delays_weather_sum']]
dat_weather.sort_values(by=['delays_weather_sum'], inplace=True, ascending=False)
dat_weather.reset_index(drop=True, inplace=True)
Markdown(dat_weather.head(5).to_markdown(index=False))
# # calculate the ratio of weather delays to total number of delays
# dat_weather['delays_weather_avg'] = dat_weather.eval('delays_weather_sum /num_of_delays_total')
# # clean up the data
# dat_weather = dat_weather.round({'delays_weather_avg':3})
# dat_weather = dat_weather.round({'delays_weather_sum':0})
# dat_weather.reset_index(drop=True, inplace=True)
# dat_weather.sort_values(by=['delays_weather_avg'], inplace=True, ascending=False)