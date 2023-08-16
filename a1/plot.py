import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('b.csv')
# df.drop(['slow', 'low_cpu', 'hk'], axis=1)
print(df)

plt.bar(df)
plt.show()

# plt.bar(courses, values, color ='maroon',
#         width = 0.4)

# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
# plt.show()