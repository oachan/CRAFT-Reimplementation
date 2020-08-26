# CRAFT-Reimplementation
給特定專案使用的修改版本.

## 修改項目

### basenet/vgg16_bn.py
1. line 9 & line 21: 改成vgg16 network的資料夾位置

### data_loader.py
1. `if random.random() > 0.01 and len(word_bboxes) > 0:`: random crop 的閥值
  - ICDAR_2015: 0.6
  - MLR       : 0.35
  - SciTSR    : 0.01 (自己設定的)
  
2. `gt_path = os.path.join(self.gt_folder, "%s.txt" % os.path.splitext(imagename)[0])`: 更改輸入txt的檔名

3. `class PRL5fold(craft_base_dataset):`

  - 自己寫的dataloader, 可用於PRL以及SciTSR
  - 參照 class ICDAR_2015 下去修改
  - 額外新增label錯誤的檢測功能

### file_utils.py

1. 修改`saveResult()`函式儲存結果的方法及位置, 不過沒有使用到

### test.py

1. `trained_model`: 已經訓練好的模型

2. `canvas_size`: 輸入圖片會照比例resize成canvas_size (好像必須為32的倍數)

3. line 61 ~ line 65: 輸入圖片資料夾 & 輸出結果資料夾

4. `with torch.no_grad():`(line 81): 不加的話, GPU內存會爆掉

### train_PRL_data.py

1. 參考 `trainic15data.py` 的格式下去寫的
