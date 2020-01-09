# fast style transfer

利用一張特定風格的圖片加上 COCO 的訓練資料集訓練模型，訓練完輸入一張圖片即輸出風格轉變後的結果，神經網路根據《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》設計

![](https://i.imgur.com/8LVtABR.png)

利用VGG16預先訓練好的模型取 features 來計算 Loss，其中分為：
1. content loss: 利用VGG16 relu3_3層輸出的特徵，比對原圖與輸出圖片
2. style loss: 利用VGG16 relu1_2, relu2_2, relu3_3, relu4_3層輸出的特徵，比對原圖與特定風格的圖片

最後利用 content loss + style loss 來訓練網路

由於神經網路比較多層加上 dataset 有將近 8G 的資料用我的桌機 train 了兩天都沒好，所以先用網路上其他人 train 的模型

風格圖片：
![image](https://i.imgur.com/knbV6es.jpg =60%x)

原圖：
![image](https://i.imgur.com/u7dujRP.jpg =45%x) ![image](https://i.imgur.com/wpNxPw2.jpg =45%x)

結果：
![image](https://i.imgur.com/1GetiHD.jpg =45%x) ![image](https://i.imgur.com/9MAt1wJ.jpg =45%x) 

看起來顏色變化太過平坦的區域太廣結果看起來會怪怪的xd