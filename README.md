
<div align="center">
<h1 align="center">🛰️GUSO Dataset and FHReg Method🛰️</h1>

<h3>Ultra-high-Resolution SAR and Optical Image Registration: From Global Benchmark Dataset to Frequency-Guided Registration Method</h3>


[Heng Yan](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>a</sup>, [Ailong Ma](https://scholar.google.com/citations?user=GPjZ_2gAAAAJ&hl=en)<sup>a *</sup>, Hong Shu<sup>a</sup>, [Yuting Wan](https://scholar.google.com/citations?user=BkQSQ6wAAAAJ&hl=en)<sup>a</sup>, [Liangpei Zhang](https://scholar.google.com/citations?user=yFEl8hcAAAAJ&hl=en)<sup>a</sup>, [Yanfei Zhong](https://scholar.google.com/citations?user=Fm7XZ5AAAAAJ&hl=en)<sup>a</sup>  

<sup>a</sup> State Key Laboratory of Information Engineering in Surveying, Mapping, and Remote Sensing, Wuhan University

</div>

## 🛎️Updates
* **` Mar 09th, 2026`**: Our Researcher paper has been accepted by ***ISPRS Journal of Photogrammetry and Remote Sensing*** and [online available](https://www.sciencedirect.com/science/article/pii/S0924271626001243) now!
* **` Jan 30th, 2026`**: GUSO dataset, FHReg pre-trained weight, and test demo have been released!
* **` Jan 13th, 2026`**: GUSO dataset and FHReg test demo are coming soon!


## ✨ Highlights

* 🌎**Global Scale**: Comprising `593,335` UHR SAR-optical image pairs (`0.16–0.98 m` resolution) spanning `319` cities across `78` countries and `six` continents.
* 🎯**Precision Alignment**: Achieving rigorous spatial consistency through fine-grained manual registration, meeting the demands of precision cross-modal applications.
* 💎**Broad Generalizability**: Beyond `multi-modal remote sensing image registration`, it serves as a robust benchmark for diverse downstream tasks: `multi-modal image fusion` and `foundation model pre-training and fine-tuning`.
* 💪**Robust Method**: Introducing a `frequency-guided hierarchical registration method`, validated across three benchmarks: the `OSdataset`, `MSAW`, and `GUSO datasets`.
---

<p align="center">
  <img src="./figure/overall.png" alt="Overall of GUSO Dataset" width="97%">
</p>


## 📥 Data Download
* To download the dataset, please fill out this [Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdCM2RUllOLOzTVKYcnxwTH9VrQrxtIXhG-YLLciy0reDmSqg/viewform?usp=publish-editor). 
* Note that the download link to the five general scene datasets will appear immediately after form submission.
```
GUSO Dataset
│
├── Urban
│    ├──urban-train.zip (186.66 GB)
│    ├──urban-val.zip (19.78 GB)
│    ├──urban-test.zip (20.04 GB)
├── Rural
│    ├──rural-train.zip (74.22 GB)
│    ├──rural-val.zip (2.93 GB)
│    ├──rural-test.zip (4.65 GB)
│── Plain
│    ├──plain-train.zip (46.73 GB)
│    ├──plain-val.zip (7.76 GB)
│    ├──plain-test.zip (7.34 GB)
├── Hill
│    ├──hill-train.zip (16.19 GB)
│    ├──hill-val.zip (1.04 GB)
│    ├──hill-test.zip (1.13 GB)
├── Water
│    ├──water-train.zip (17.91 GB)
│    ├──water-val.zip (2.24 GB)
│    ├──water-test.zip (1.48 GB)
└── Disaster
     ├──disaster-test.zip (3.35 GB)
```

## 📥 Weight Download
* The GUSO-based FHReg pre-trained weight is available for download via this [link](https://drive.google.com/drive/folders/1T1Ic0uxSIBx_qdQJQEV68jzGlwBFR3TU?usp=drive_link). This pre-trained weight enables zero-shot cross-domain evaluation on the [OSdataset](https://drive.google.com/file/d/16xUHXHJceL-c-lWBFiZ--88fbBc-WW8h/view?usp=drive_link) and [MSAW](https://drive.google.com/file/d/1yKGNJIuUicwMpoPUReGlwu-acUODWOj5/view?usp=drive_link) datasets.

  
## 📜Citation
If you use GUSO and FHReg in your research, please cite our paper.
```text
   @Article{YAN2026GUSOFHReg,
    AUTHOR = {Heng Yan and Ailong Ma and Hong Shu and Yuting Wan and Liangpei Zhang and Yanfei Zhong},
    TITLE = {Ultra-high-resolution SAR and optical image registration: From global benchmark dataset to frequency-guided registration method},
    JOURNAL = {ISPRS Journal of Photogrammetry and Remote Sensing},
    PAGES = {190-210},
    VOLUME = {235},
    YEAR = {2026},
    DOI = {10.1016/j.isprsjprs.2026.03.010}
}
    
```


## 🤝Acknowledgments
The authors would like to thank [ICEYE](https://www.iceye.com/resources/datasets), [Capella Space](https://www.capellaspace.com/earth-observation/gallery), and [Umbra Space](https://umbra.space/open-data/) for making their valuable SAR data available through their respective open data initiatives. We also thank [BRIGHT](https://github.com/ChenHongruixuan/BRIGHT/tree/master) for the provision of disaster labels.


## 🛠️Copyright
The copyright belongs to Intelligent Data Extraction, Analysis and Applications of Remote Sensing ([RSIDEA](http://rsidea.whu.edu.cn/)) academic research group, State Key Laboratory of Information Engineering in Surveying, Mapping, and Remote Sensing (LIESMARS), Wuhan University. The GUSO dataset can be used for academic purposes only and need to cite the following paper, <font color="red"><b> but any commercial use is prohibited.</b></font> Otherwise, RSIDEA of Wuhan University reserves the right to pursue legal responsibility.

## 🙋Q & A
***For any questions, please [contact us](mailto:yanheng0903@gmail.com) at **yanheng0903@gmail.com**.***




