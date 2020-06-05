#作業1 撰寫MSE函數
import numpy as np 
import matplotlib.pyplot as plt

def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

def mean_squared_error(y,yp):
    mse = MSE = sum((y-yp)**2) / len(y)
    return mse 

w = 3
b = 0.5
x_lin = np.linspace(0, 100, 101)
y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()

y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()

MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))

"""
作業2 kaggle URL "https://www.kaggle.com/gomes555/road-transport-brazil"
1.你選的這組資料為何重要
    交通問題在各國都是相當重要的議題之一。
2.資料從何而來
    收集自巴西國家陸路交通局，為開放性資料。
3.蒐集而來的資料型態為何
    共88964筆資料，14欄屬性，包含字串.浮點數及整數
4.這組資料想解決的問題如何評估
    藉由資料用最具創意的方式介紹巴西旅遊行程

作業3 
1.核心問題
    1希望目標 :提升整體營業額（載客量越多越好.時間越短越好（盡量避開塞車路段））
2.資料及如何蒐集
    2.1蒐集各時間點不同地區人潮數量
        2.1.1觀光局有提供各知名觀光景點月人潮及成長率等資料　
        URL:"https://admin.taiwan.net.tw/FileUploadCategoryListC003330.aspx?CategoryID=3afda5d8-b1ac-4bf4-b732-d2f35e37c7f2&appname=FileUploadCategoryListC003330"
    2.2蒐集各時間乘客需求類別（旅遊.出差等等......）以及出發地與目的地
        2.2.1可設計APP提供叫車服務，並記錄乘客上下車地點。另外可鼓勵乘客填寫簡易乘坐資料及評分，給予額外乘車折扣或禮卷
    2.3各時段不同地點車流量資料
        2.3.1交通部公路總局可查詢公開各路段車流當日資料，可分析出各路段易塞車時段以規避
        URL:"https://www.thb.gov.tw/sites/ch/modules/download/download_list?node=bcc520be-3e03-4e28-b4cb-7e338ed6d9bd&c=83baff80-2d7f-4a66-9285-d989f48effb4"
3.蒐集而來的資料型態為何
    應該大多也是字串.浮點數跟整數
4.如何評估
    觀查月營業額變化
"""