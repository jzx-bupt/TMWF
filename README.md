# Transformer-based Model for Multi-tab Website Fingerprinting Attack

## Introduction

This repository presents the complete version of the paper "Transformer-based Model for Multi-tab Website Fingerprinting Attack" in ACM CCS 2023 along with corresponding codes and datasets.  

If you want to complete the "synthesize multi-tab traces, train the model, and evaluate" process, after running the "MergeSingleTraces_openworld.py" file to generate the synthetic traces, you may need to manually rename some files in order to read by subsequent program.  

We apologize for any inconsistencies in the code formatting, as these codes are intended solely for research purposes. Your understanding is greatly appreciated while reviewing them.  

If these data and code contribute to your work, please cite this paper, and here is the corresponding bibtex text:  

```
@inproceedings{10.1145/3576915.3623107,
author = {Jin, Zhaoxin and Lu, Tianbo and Luo, Shuang and Shang, Jiaze},
title = {Transformer-Based Model for Multi-Tab Website Fingerprinting Attack},
year = {2023},
isbn = {9798400700507},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3576915.3623107},
doi = {10.1145/3576915.3623107},
abstract = {While the anonymous communication system Tor can protect user privacy, website fingerprinting (WF) attackers can still identify the websites that users access over encrypted network connections by analyzing the metadata generated during network communication. Despite the emergence of new WF attack techniques in recent years, most research in this area has focused on pure traffic traces generated from single-tab browsing behavior. However, multi-tab browsing behavior significantly degrades the performance of WF classification models based on the single-tab assumption. As a result, some research has shifted its focus to multi-tab WF attacks, although most of these works have limited utilization of the mixed information contained in multi-tab traces. In this paper, we propose an end-to-end multi-tab WF attack model, called Transformer-based model for Multi-tab Website Fingerprinting attack (TMWF). Inspired by object detection algorithms in computer vision, we treat multi-tab WF recognition as a problem of predicting ordered sets with a maximum length. By adding enough single-tab queries to the detection model and letting each query extract WF features from different positions in the multi-tab traces, our model's Transformer architecture capitalizes more fully on trace features. Paired with our new proposed model training approach, we accomplish adaptive recognition of multi-tab traces with varying numbers of web pages. This approach successfully eliminates a strong and unrealistic assumption in the field of multi-tab WF attacks - that the number of tabs contained in a sample belongs to the attacker's prior knowledge. Experimental results in various scenarios demonstrate that the performance of TMWF is significantly better than existing multi-tab WF attack models. To evaluate model performance in more authentic scenarios, we present a dataset of multi-tab trace data collected from real open-world environments.},
booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
pages = {1050–1064},
numpages = {15},
keywords = {privacy, multi-tab website fingerprinting, transformer, tor},
location = {<conf-loc>, <city>Copenhagen</city>, <country>Denmark</country>, </conf-loc>},
series = {CCS '23}
}
```

## File Description
    ├── code  #Part of the codes used in the experiments                     
        ├── MergeSingleTraces_openworld.py: Synthesize multi-tab traces through the original single-tab traces. The JSON files generated alongside the multi-tab trace dataset contain annotations, such as the starting and ending points of individual page segments. Although we used this data while debugging the DETR source code earlier, it is not involved in the current TMWF code experiments.
        ├── metrics.py: All evaluation metrics used in the experiments        
        ├── models.py : TMWF's impletantion by pytorch
        ├── real_openworld.py: Codes of real openworld experiments, use real multi-tab traces  
        └── simulation_openworld.py: Codes of simulation openworld experiments, use synthesized multi-tab traces
    ├── dataset  #Part of the datasets used in the experiments, use Python Pickle module to load. Please be aware that due to the imperfections in our code implementation, manual renaming of certain generated files may be necessary after manually synthesizing multi-tab trace datasets to facilitate program loading.
        ├── chrome realworld: Except for "2tabs-extra" containing over 10,000 2-tab chrome browser trace samples, each file contains over 1,000 samples
        ├── chrome single tab: 5,000 monitored traces (comprising 50 monitored websites, each with 100 individual single-tab traces), along with 5,000 non-monitored traces, were stratified and partitioned into training and testing (or validation) sets in a 4:1 ratio, used for synthesizing multi-tab traces
        ├── tbb realworld: Except for "2tabs-extra" containing over 10,000 2-tab tor browser trace samples, each file contains over 1,000 samples
        └── tbb single tab: 5,000 monitored traces (comprising 50 monitored websites, each with 100 individual single-tab traces), along with 5,000 non-monitored traces, were stratified and partitioned into training and testing (or validation) sets in a 4:1 ratio, used for synthesizing multi-tab traces      
    ├── complete_paper_version_of_TMWF.pdf      
    └── README.md

## Supplementary Description
Due to constraints such as the length of conference papers, thematic requirements, and other limiting factors, we have omitted certain content in the final paper draft. We summarize these ideas and potential research directions below:  
* Due to the fact that multi-class models cannot ignore the differences between various website classes within the monitored set, as well as on our experimental results and the conclusions in recent works [1](https://www.usenix.org/conference/usenixsecurity22/presentation/cherubin), it may be challenging for current WF classifiers to perform effectively in real open-world scenarios with end-to-end multi-classification tasks. However, breaking down this end-to-end task into binary and multi-classification sub-tasks (as well as described in recent works [2](https://ieeexplore.ieee.org/document/9731352) [3](https://petsymposium.org/popets/2023/popets-2023-0125.php)) could potentially be a method for performance improvement.  
* Sliding window mechanism: By setting a window size (i.e., the model's recognition limit and the segment length within the window) and the overlap ratio between windows, the attacker can merge the recognition results of multiple local segments to determine the recognition result of the complete trace. Although intuitively, the sliding window mechanism can be used to extend single-tab WF attacks to multi-tab traces, the overlapping parts of multi-tab traces may lead to more erroneous results from these classifiers. As for how to choose the parameters of the sliding window, we believe it involves a trade-off to maximize efficiency and performance, which is a topic for future exploration.  
* In the experiment, we only used a training set with a single fixed N-value setting (the maximum number of pages the model supports to recognize). According to the opinions of anonymous reviewers, researchers can try to match a variety of different page number multi-tab trace samples in the training set in the future to explore the performance changes of the model.  

**Reference:**  
[1] Cherubin, Giovanni, Rob Jansen, and Carmela Troncoso. "Online website fingerprinting: Evaluating website fingerprinting attacks on Tor in the real world." 31st USENIX Security Symposium (USENIX Security 22). 2022.  
[2] Wang, Yanbin, et al. "SnWF: Website fingerprinting attack by ensembling the snapshot of deep learning." IEEE Transactions on Information Forensics and Security 17 (2022): 1214-1226.  
[3] Jansen, Rob, and Ryan Wails. "Data-Explainable Website Fingerprinting with Network Simulation." Proceedings on Privacy Enhancing Technologies 4 (2023): 559-577. 

## Future Directions
In our future work, we contemplate the following potential research directions:  
1. Incorporating transfer learning by utilizing a pre-trained backbone to enhance the efficiency of our model attacks.  
2. Expanding the task scenarios to investigate the model's performance under various conditions, including concept drift, the base rate of different websites, WF defense technologies, etc.  
3. Attempting data augmentation techniques, inspired by studies in the field of computer vision, to automatically modify the original trace sequences to simulate noise in real-world environments.  

We hope these intuitions and potential research directions can be pursued in future work.

## Appeal
We would like to emphasize that due to the relatively limited hardware resources of the VPS we used in our experiments (with a maximum of only a dual-core processor and 2.5GB of RAM), we frequently encountered access failures when configuring the Tor Browser Selenium driver (http://github.com/webfp/tor-browser-selenium). These failures were often due to hardware performance bottlenecks, and such incidents may have led to reduced trustworthiness of the datasets we collected. Therefore, we encourage researchers to employ VPS with higher performance capabilities to simulate Tor clients and collect larger-scale datasets.

## Contacts
Zhaoxin Jin, jzx3990@bupt.edu.cn  
Shuang Luo, lok@bupt.cn