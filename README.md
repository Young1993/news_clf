## active learning

Problem: cannot understand the number extracted by jieba \
- step 1: select 15 sample, then train classifier, choose top 5 best and 10 worst
- predict active sample  \
｜1- 5 round: query hard 15 sample find the top 5 highest probability and the bottom 10 worst probability data
｜after 6 round top 1 and bottom 14;
｜12-15 round top 1 amd bottom 24 \
关键问题：1、关键词筛选；
2、hard_sample 筛选
-----
- step 2: repeat the procedure


