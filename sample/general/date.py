import datetime, time

def getTimeSpan(timespan):
    dateTime = time.localtime(timespan / 1000)
    dateFormat = time.strftime("%Y-%m-%d %H:%M:%S", dateTime)
    return dateFormat


if __name__ == "__main__":
    dateFormat = getTimeSpan(1677573715720)
    print(dateFormat)
    time = round(time.mktime(time.localtime()))
    print(time)