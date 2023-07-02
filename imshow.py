from sample.paper.paper import Imshow
import pandas as pd
from sample.paper.map import MAP
import numpy as np
from sample.general.imshow import Bar

if __name__ == "__main__":
    # data = pd.read_excel('c:/users/80521/desktop/imshow.xlsx')
    # figure 3-1, 3-2, 3-3, 3-5 code.
    # Imshow.imshow_3_1(data)
    # Imshow.imshow_3_2(data)
    # Imshow.imshow_3_3(data)
    # Imshow.imshow_3_5(data)

    # figure 3-4, 3-6 code.
    # name_map = {
    # '阿克苏地区':'阿克苏','阿勒泰地区':'阿勒泰','巴音郭楞蒙古自治州':'巴州',
    # '博尔塔拉蒙古自治州':'博州','昌吉回族自治州':'昌吉','哈密市':'哈密','和田地区':'和田',
    # '喀什地区':'喀什','克拉玛依市':'克拉玛依','克孜勒苏柯尔克孜自治州':'克州','石河子市':'石河子',
    # '塔城地区':'塔城','吐鲁番市':'吐鲁番','乌鲁木齐市':'乌鲁木齐','伊犁哈萨克自治州':'伊犁','铁门关市':'铁门关',
    # '阿拉尔市':'阿拉尔','图木舒克市':'图木舒克','可克达拉市':'可克达拉','北屯市':'北屯','双河市':'双河',
    # '昆玉市':'昆玉','五家渠市':'五家渠'
    # }
    # data = pd.read_excel('c:/users/80521/desktop/无作者/imshow___.xlsx')
    # arrayAttention = np.array(data.reindex(columns = ["district", "efficient"]))
    # attentionRegion = arrayAttention.tolist()
    # noAttentionRegion = [["石河子", 0], ["铁门关", 0], ["阿拉尔", 0], ["图木舒克", 0], ["可克达拉", 0], ["北屯", 0], ["双河", 0], ["昆玉", 0], ["五家渠", 0]]
    # MAP.createCityMap(name_map, attentionRegion, noAttentionRegion) 
    bar = Bar("bar figure")
    bar.imshow()
    """     
    data = pd.read_excel('c:/users/80521/desktop/无作者/imshow.xlsx')
    Imshow.imshow_3_6(data)  
    """
    # data = pd.read_excel('c:/users/80521/desktop/无作者/imshow_.xlsx')
    # Imshow.imshow_3_7(data)  