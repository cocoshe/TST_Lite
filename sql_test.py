import numpy as np
import pymysql


db = pymysql.Connect(host='localhost', port=3306, user='root', passwd='123123', db='yuheng', charset='utf8')
cursor = db.cursor()

sql = 'select threshold from threshold'
res = cursor.execute(sql)
res = np.array(cursor.fetchall())[:, 0]
print('res1:', res)

cursor.execute('select * from company')
res = cursor.fetchall()
print('res2:', res)

# test = np.array([10, 20, 30, 40, 50])
# chose = np.where(test < 20)
# print('chose:', chose[0])