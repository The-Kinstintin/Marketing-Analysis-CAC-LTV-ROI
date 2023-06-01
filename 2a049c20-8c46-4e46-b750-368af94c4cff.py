#!/usr/bin/env python
# coding: utf-8

# ### Инструкция по выполнению проекта:
# 
# Вы - маркетинговый аналитик развлекательного приложения Procrastinate Pro+. Несколько прошлых месяцев ваш бизнес постоянно нес убытки - в привлечение пользователей была вложена куча денег, а толку никакого. Вам нужно разобраться в причинах этой ситуации.
# 
# У вас в распоряжении есть лог сервера с данными о посещениях приложения новыми пользователями, зарегистрировавшимися в период с 2019-05-01 по 2019-10-27, выгрузка их покупок за этот период, а также статистика рекламных расходов. Вам предстоит изучить, как люди пользуются продуктом, когда они начинают покупать, сколько денег приносит каждый клиент, когда он окупается и какие факторы отрицательно влияют на привлечение пользователей.
# 
# #### Шаг 1. Загрузите данные и подготовьте их к анализу
# Загрузите данные о визитах, заказах и расходах в переменные. Оптимизируйте данные для анализа. Убедитесь, что тип данных в каждой колонке — правильный. Путь к файлам:
# 
#  -   /datasets/visits_info_short.csv. Скачать датасет
#  -   /datasets/orders_info_short.csv. Скачать датасет
#  -   /datasets/costs_info_short.csv. Скачать датасет
#  
# #### Шаг 2. Задайте функции для расчета и анализа LTV, ROI, удержания и конверсии
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.
# 
# #### Шаг 3. Проведите исследовательский анализ данных
# 
# Постройте профили пользователей. Определите минимальную и максимальную дату привлечения пользователей.
# 
# Выясните:
# - Из каких стран приходят посетители? Какие страны дают больше всего платящих пользователей?
# - Какими устройствами они пользуются? С каких устройств чаще всего заходят платящие пользователи?
# - По каким рекламным каналам шло привлечение пользователей? Какие каналы приносят больше всего платящих пользователей?.
# 
# #### Шаг 4. Маркетинг
# Выясните:
# 
# - Сколько денег потратили? Всего / на каждый источник / по времени
# - Сколько в среднем стоило привлечение одного покупателя из каждого источника?
# 
# #### Шаг 5. Оцените окупаемость рекламы для привлечения пользователей
# 
# С помощью LTV и ROI:
# - Проанализируйте общую окупаемость рекламы;
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам;
# - Проанализируйте окупаемость рекламы с разбивкой по странам;
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам.
# 
# Опишите проблемы, которые вы обнаружили. Ответьте на вопросы:
# - Окупается ли реклама, направленная на привлечение пользователей в целом? 
# - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
# - Чем могут быть вызваны проблемы окупаемости? Изучите конверсию и удержание с разбивкой по устройствам, странам, рекламным каналам.
# 
# Опишите возможные причины обнаруженных проблем и сформируйте рекомендации для рекламного отдела. При решении этого шага считайте, что вы смотрите данные 1-го ноября 2019 года и что в вашей организации принято считать, что окупаемость должна наступать не позднее, чем через 2 недели после привлечения пользователей.
# 
# Подумайте, нужно ли включать в анализ органических пользователей?
# 
# #### Шаг 6. Напишите выводы
# - Выделите причины неэффективности привлечения пользователей;
# - Сформируйте рекомендации для отдела маркетинга для повышения эффективности.
# 
# #### Оформление: 
# Задание выполните в Jupyter Notebook. Программный код заполните в ячейках типа code, текстовые пояснения — в ячейках типа markdown. Примените форматирование и заголовки.
# 
# 
# #### Описание данных
# Таблица visits_log_short (лог сервера с информацией о посещениях сайта):
# 
#     User Id — уникальный идентификатор пользователя
#     Device — категория устройства пользователя
#     Session start — дата и время начала сессии
#     Session End — дата и время окончания сессии
#     Channel — идентификатор рекламного источника, из которого пришел пользователь
#     Region - страна пользователя
# 
# Таблица orders_log_short (информация о заказах):
# 
#     User Id — уникальный id пользователя, который сделал заказ
#     Event Dt — дата и время покупки
#     Revenue — выручка
# 
# Таблица costs_short (информация о затратах на маркетинг):
# 
#     Channel — идентификатор рекламного источника
#     Dt — дата
#     Costs — затраты на этот рекламный источник в этот день

# # Шаг 1. Загрузите данные и подготовьте их к анализу
# Загрузите данные о визитах, заказах и расходах в переменные. Оптимизируйте данные для анализа. Убедитесь, что тип данных в каждой колонке — правильный. Путь к файлам:
# 
#  -   /datasets/visits_info_short.csv. 
#  -   /datasets/orders_info_short.csv. 
#  -   /datasets/costs_info_short.csv.

# In[1]:


import pandas as pd
pd.set_option('max_columns', 100)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


visits = pd.read_csv('/datasets/visits_info_short.csv')
orders = pd.read_csv('/datasets/orders_info_short.csv')
costs = pd.read_csv('/datasets/costs_info_short.csv')


# In[3]:


sets = [visits,orders,costs]
set_name = ['Visits','Orders','Costs']

def show_sets (dfs):
    for i in range (0, len(dfs)):
        print('')
        print('============================================================================================')
        print('')
        print( set_name[i],'data overview' )
        display(dfs[i].head())

        print('')
        print(set_name[i], 'size of DataFrame:', dfs[i].shape)
        print('')

        print( set_name[i], 'check the Dtype and Null values')
        dfs[i].info()
        print('')
        print('Number of dulplicates:', len(dfs[i][dfs[i].duplicated()==True]))

show_sets(sets)        


# First observation has showed no null values and no duplicates in data. 
# 
# Our next step will be transform data sets. We will change the name of columns to one standard (lower case, underscore) and change date columns to dt format
# 
# +++ Kostyan pomenyai cvet +++

# In[4]:


for a in sets:
    
    col = list(a.columns)
    
    new_col = [] 
    
    for i in a:
    
        i = i.lower().replace(' ','_')
        new_col += [i]  
    a.columns = new_col
    display(a.head())

visits['session_start'] = visits['session_start'].astype('datetime64[ns]')
visits['session_end'] = visits['session_end'].astype('datetime64[ns]')
orders['event_dt'] = orders['event_dt'].astype('datetime64[ns]')
costs['dt'] = pd.to_datetime(costs['dt']).dt.date


# After all changes lets recheck all data sets

# In[5]:


show_sets(sets)
    


# Data clearence and transformation is compleeted. It is a time to start our analysis

# EDA analysis of datasets:

# Fisrt we analyse visits.

# Check of Time properties of visits. 
# First time window

# In[6]:


visits = visits.sort_values(by='session_start', ascending=True)
display(visits.head(5))


# We can see that in our dataset we have 6 complete months from 1.st of May untill end of October. 
# 
# Next step we will check the session duration and clean 0 s. visits. 

# In[7]:


visits['ses_len_min'] = (visits['session_end'] - visits['session_start'])
visits['ses_len_min'] = round(visits['ses_len_min'].dt.total_seconds()/60,2)

display(visits.query('ses_len_min == 0.0'))

visits = visits.query('ses_len_min != 0.0')



# We have find out that in our log there are 163 sessions with 0 duration. It is a less then a 0.1% of our total data, probably no purchases made during these sessions. Therefore, we drop them as irrelevant for the EDA analysis. However, this sessions could be checked by data engenieers.

# In[8]:


display(visits['ses_len_min'].describe())
print('')
print('Median:',visits['ses_len_min'].median())
print('90-95-99 percentile:', np.percentile(visits['ses_len_min'],[90,95,99]))


# In[9]:


plt.figure(figsize=(12,6))
plt.hist(visits["ses_len_min"], bins = 1000)
plt.axvline(x = visits['ses_len_min'].mean() , color = 'b', label = 'mean')
plt.axvline(x = visits['ses_len_min'].median() , color = 'r', label = 'median')
plt.legend(loc='upper right')
plt.show()


# Average duration is 30 min, Meadian is 20 min. Standard deviation 30. The 10 % of longest sessions took over 1h 10m  

# ===== Second we analyze devices. To make full analysis first we merge visits and orders to identify people who completed purchases. Then we will find the users and customers ratio of channels and regions =====

# Check if any users counted in more channels or regions

# In[10]:


visits['customer'] = visits['user_id'].isin(orders['user_id'].unique()).astype(int)
display(visits.head())


# In[11]:


visits_region = visits.groupby(by=['region'], as_index= False).agg({'user_id':'nunique'})
visits_customers = visits.query('customer == 1').groupby(by=['region'], as_index= False).agg({'user_id':'nunique'})


# In[12]:


customer_ratio = visits_region.merge(visits_customers, on ='region')
customer_ratio.columns =['region', 'users','customers']
customer_ratio['%'] = customer_ratio['customers']/customer_ratio['users'] 
display(customer_ratio)


# In[13]:


print('More then 1 region:', visits.groupby(by='user_id').agg({'region':'nunique','channel':'nunique'}).query('region > 1').shape[0])
print('More then 1 channel:', visits.groupby(by='user_id').agg({'region':'nunique','channel':'nunique'}).query('channel > 1').shape[0])
print('More then 1 device:', visits.groupby(by='user_id').agg({'region':'nunique','device':'nunique'}).query('device > 1').shape[0])


# In[14]:


user_channel = visits.groupby(by='user_id', as_index=False).agg({'channel':'first'})
user_channel.columns = ['user_id','first_channel']
visits = visits.merge(user_channel, on='user_id')

user_device = visits.groupby(by='user_id', as_index=False).agg({'device':'first'})
user_device.columns = ['user_id','first_device']
visits = visits.merge(user_device, on='user_id')



display(visits.head(10))


# In[15]:


users = visits.groupby('user_id', as_index=False).agg({'region':'first','first_device':'first'})
customers = visits.query('customer == 1').groupby('user_id', as_index=False).agg({'region':'first','first_device':'first'})

visits_device_1 = users.groupby(by=['region', 'first_device'], as_index= False).agg({'user_id':'count'})
visits_device_1 = visits_device_1.merge(visits_region, on='region')
visits_device_1.columns = ['region','first_device', 'devices_count','region_users']
visits_device_1['% of devices'] = round(visits_device_1['devices_count']/visits_device_1['region_users'],4)*100


visits_device_2 = customers.groupby(by=['region', 'first_device'], as_index= False).agg({'user_id':'count'})
visits_device_2 = visits_device_2.merge(visits_region, on='region')
visits_device_2.columns = ['region','first_device', 'customers','region_users']
visits_device_2['% of customers'] = round(visits_device_2['customers']/visits_device_2['region_users'],4)*100


# In[16]:


device = list(visits_device_1['first_device'].unique())


plt.figure(figsize=(9,9))
for i in range (0, len(device)):
    a = device[i]
    temp_1 = visits_device_1.query('first_device == @a ')
    temp_2 = visits_device_2.query('first_device == @a ')
    s = int(str(22)+str(i+1))
    display(temp_1)
    display(temp_2)
    plt.subplot(s)
    plt.title(a)
    plt.bar(temp_1["region"], temp_1['% of devices'], alpha = 0.5)
    plt.bar(temp_2["region"], temp_2['% of customers'], alpha = 0.5)

plt.show()


# Regions are ok, however channels included missinformation about users history. For ROI and CAC analysis we need to clear the data.
# Now, as we dont have instructions about channels priority  we rewrite in visits channel to the first occured channel by user 

# In[17]:


visits_region_channel = visits.groupby(by=['region','first_channel'], as_index=False).agg({'user_id':'nunique'})
visit_region_customer = visits.query('customer == 1').groupby(by=['region','first_channel'], as_index=False).agg({'user_id':'nunique','region':'first','first_channel':'first'})

visits_region_channel= visits_region_channel.merge(visit_region_customer, on=['region','first_channel'])
visits_region_channel= visits_region_channel.merge(visits_region, on='region')
visits_region_channel.columns = ['region','first_channel','users','customers','region_users']
visits_region_channel['users%_of_region'] = round(visits_region_channel['users']/visits_region_channel['region_users'],4)*100
visits_region_channel['customer%_of_channel'] = round(visits_region_channel['customers']/visits_region_channel['region_users'],4)*100
region = list(visits_region_channel['region'].unique())

#display(visits_region_channel)
for i in region:
    display(visits_region_channel.query('region == @i '))


# In[18]:


plt.figure(figsize=(18,12))


for i in range (0, len(region)):
    a = region[i]
    temp_1 = visits_region_channel.query('region == @a ')
    s = int(str(22)+str(i+1))
    display(temp_1)
    plt.subplot(s)
    plt.title(a)
    plt.bar(temp_1["first_channel"], temp_1['users%_of_region'], alpha = 0.5)
    plt.bar(temp_1["first_channel"], temp_1['customer%_of_channel'], alpha = 0.5)
    plt.xticks(rotation=11)

    
plt.show()


# Users' and Customers' ratio among European countris are similar, difference in 1% range. Organic part  has simmilar customer ratio also in US, with a slighty better percentage. All countries have 6 channel to generate new_users. 
# 
# Let's check the devices distribution 

# On the next step we will check the distribution of orders amount and revenue per user.

# In[19]:


temp = orders.groupby('user_id', as_index = False).agg({'event_dt':'count','revenue':'sum'})
display(temp[['event_dt','revenue']].describe().T)

plt.hist(temp['event_dt'])
plt.show()

plt.hist(temp['revenue'])
plt.show()


# Calculate CAC and ROI:

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


display(costs.head())
display(visits.head())


# In[21]:


profiles = (
        visits.sort_values(by=['user_id', 'session_start'])
        .groupby('user_id')
        .agg(
            {
                'session_start': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first',
            }
        )
         # время первого посещения назовём first_ts
        .rename(columns={'session_start': 'first_ts'})
        .reset_index()  # возвращаем user_id из индекса
    )
profiles['dt'] = profiles['first_ts'].dt.date
profiles['month'] = profiles['first_ts'].astype('datetime64[M]')


# In[22]:


new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
         # столбец с числом пользователей назовём unique_users
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()  # возвращаем dt и channel из индексов
    )


# In[23]:


ad_costs = costs.merge(new_users, on=['dt', 'channel'], how='left')


# In[24]:


ad_costs['acquisition_cost'] = ad_costs['costs'] / ad_costs['unique_users']


# In[25]:


profiles = profiles.merge(
    ad_costs[['dt', 'channel', 'acquisition_cost']],
    on=['dt', 'channel'],
    how='left',
)


# In[26]:


profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)


# In[27]:


profiles.pivot_table(
    index='dt', columns='channel', values='acquisition_cost', aggfunc='mean'
).plot(grid=True, figsize=(10, 5))
plt.ylabel('CAC, $')
plt.xlabel('Дата привлечения')
plt.title('Динамика САС по каналам привлечения')
plt.show()


# In[ ]:





# In[28]:


def get_ltv(
    profiles,  # Шаг 1. Получить профили и данные о покупках
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # Шаг 2. Добавить данные о покупках в профили

    result_raw = result_raw.merge(
        # добавляем в профили время совершения покупок и выручку
        purchases[['user_id', 'event_dt', 'revenue']],
        on='user_id',
        how='left',
    )

    # Шаг 3. Рассчитать лайфтайм пользователя для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):

        # Шаг 4. Построить таблицу выручки
        # строим «треугольную» таблицу
        result = df.pivot_table(
            index=dims,
            columns='lifetime',
            values='revenue',  # в ячейках — выручка за каждый лайфтайм
            aggfunc='sum',
        )

        # Шаг 5. Посчитать сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)

        # Шаг 6. Вычислить размеры когорт
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )

        # Шаг 7. Объединить размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)

        # Шаг 8. Посчитать LTV
        # делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_days))]
        # восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу LTV
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # для таблицы динамики LTV убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []
    # получаем таблицу динамики LTV
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы LTV и сырые данные
    return result_raw, result_grouped, result_in_time


# In[29]:


ltv_raw, ltv, ltv_history = get_ltv(
    profiles, orders, datetime(2019, 10, 30).date(), 14, dimensions=['channel']
)

# таблица LTV
display(ltv_raw)

# кривые LTV
report = ltv.drop(columns=['cohort_size'])
report.T.plot(grid=True, figsize=(10, 5), xticks=list(report.columns.values))
plt.title('LTV с разбивкой по источникам')
plt.ylabel('LTV, $')
plt.xlabel('Лайфтайм')
plt.show()


# In[30]:


# находим максимальную дату привлечения из сырых данных LTV
max_acquitision_dt = ltv_raw['dt'].max()
# отсекаем профили, которые «старше» этой даты
ltv_profiles = profiles.query('dt <= @max_acquitision_dt')

# оставшееся число пользователей на каждый лайфтайм
ltv_profiles.groupby('dt').agg({'user_id': 'nunique'})


# In[31]:


# считаем средний CAC по каналам привлечения

cac = (
    ltv_profiles.groupby('channel')
    .agg({'acquisition_cost': 'mean'})
    .rename(columns={'acquisition_cost': 'cac'})
)

cac


# In[32]:


roi = ltv.div(cac['cac'], axis=0)
roi


# In[33]:


display(ltv)


# In[34]:


roi.loc[:,'cohort_size'] = ltv['cohort_size']
roi


# In[35]:


report = roi.drop(columns=['cohort_size'])
report.T.plot(grid=True, figsize=(10, 5), xticks=list(report.columns.values))

plt.title('ROI с разбивкой по каналам привлечения')
plt.ylabel('ROI')
plt.xlabel('Лайфтайм')
plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Шаг 2. Задайте функции для расчета и анализа LTV, ROI, удержания и конверсии
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.

# # Шаг 3. Проведите исследовательский анализ данных
# 
# Постройте профили пользователей. Определите минимальную и максимальную дату привлечения пользователей.
# 
# Выясните:
# - Из каких стран приходят посетители? Какие страны дают больше всего платящих пользователей?
# - Какими устройствами они пользуются? С каких устройств чаще всего заходят платящие пользователи?
# - По каким рекламным каналам шло привлечение пользователей? Какие каналы приносят больше всего платящих пользователей?

# # Шаг 4. Маркетинг
# 
# Выясните:
# - Сколько денег потратили? Всего / на каждый источник / по времени
# - Сколько в среднем стоило привлечение одного покупателя из каждого источника?

# # Шаг 5. Оцените окупаемость рекламы для привлечения пользователей
# 
# С помощью LTV и ROI:
# - Проанализируйте общую окупаемость рекламы;
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам;
# - Проанализируйте окупаемость рекламы с разбивкой по странам;
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам.
# 
# Опишите проблемы, которые вы обнаружили. Ответьте на вопросы:
# - Окупается ли реклама, направленная на привлечение пользователей в целом? 
# - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
# - Чем могут быть вызваны проблемы окупаемости? Изучите конверсию и удержание с разбивкой по устройствам, странам, рекламным каналам.
# 
# Опишите возможные причины обнаруженных проблем и сформируйте рекомендации для рекламного отдела. При решении этого шага считайте, что вы смотрите данные 1-го ноября 2019 года и что в вашей организации принято считать, что окупаемость должна наступать не позднее, чем через 2 недели после привлечения пользователей.

# ### Проанализируйте общую окупаемость рекламы

# ### Проанализируйте окупаемость рекламы с разбивкой по устройствам

# ### Проанализируйте окупаемость рекламы с разбивкой по странам

# ### Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам

# # Шаг 6. Напишите выводы
# - Выделите причины неэффективности привлечения пользователей;
# - Сформируйте рекомендации для отдела маркетинга для повышения эффективности.
