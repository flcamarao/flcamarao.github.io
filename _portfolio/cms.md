---
title: "Comprehensive Medicare Data Analysis for Insurance Companies"
excerpt: "In this study, the team leveraged Big Data techniques to analyze the CMS Medicare data and provide valuable insights for insurance companies to optimize their policies and coverage options. By effectively handling the large-scale dataset, they were able to uncover patterns in healthcare service utilization, payment dynamics, and drug claims among insured policyholders, leading to actionable recommendations for improved healthcare delivery.<br /><img src='/images/cms/1.png'>"
date: December 6, 2022
collection: portfolio
---

<img src='/images/cms/1.png'>

<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">EXECUTIVE SUMMARY</h1>


In an era where healthcare costs are soaring, and the demand for efficient and high-quality care is increasing, understanding the patterns of healthcare service utilization and associated costs is of paramount importance. This study is motivated by the desire to identify opportunities for insurance companies to optimize their policies, expand coverage options, and ultimately, deliver better healthcare to Medicare beneficiaries.

To address the research question **"How can insurance companies optimize their policies and coverage options based on the analysis of CMS Medicare data?"**, the team embarked on a comprehensive exploration of the CMS Data by Provider and Service and by Provider and Drug. This endeavor aimed to uncover valuable insights into healthcare service utilization patterns, payment dynamics, submitted charges, and drug claims among insured policyholders. By navigating the complexities of this data, the team sought to provide solutions that would enable insurance companies to optimize their policies and coverage options effectively.

The methodology employed by the team can be summarized into three key sections. It began with data collection, followed by preprocessing of publicly available Centers for Medicare & Medicaid Services (CMS) datasets, and concluded with data analysis involving descriptive analytics and visualization. The team conducted a comprehensive descriptive analysis and visualization of the data, breaking it down into several sections from the distribution of service providers to the service and drug claims across different age groups. This comprehensive analysis enabled the team to delve into numerous aspects of the Medicare system, such as the role of different healthcare specialties, claims and payments, geographical differences in service provision and pricing, the impact of age factor, and the distribution and utilization of prescription drugs.

The analysis revealed significant patterns in healthcare service utilization, payment dynamics, submitted charges, and drug claims among insured policy holders. Such results underscored the significant role of primary care services, particularly `Internal Medicine` and `Family Practice`, in the healthcare delivery system. It also highlighted the high dependency on Medicare among the elderly population, with a notably higher number of claims in the `65 and above` age group. The high volume of claims for certain specialties, such as `Cardiology`, `Psychiatry`, `Neurology`, and `Nephrology`, underscored the prevalence of age-related conditions in this group.

A key finding was the dominance of `individual providers` over organizations in Medicare claims, suggesting a fragmented healthcare market within Medicare that caters to diverse patient needs. The distribution of service locations also pointed to a preference among patients for care in `offices` compared to `facilities`, aligning with the dominance of `individual providers` in Medicare.

The data further revealed that `US providers` and `non-US providers` differed significantly in their submitted charges for services, although the average Medicare payment did not significantly vary between these two groups.

These findings lay the foundation for actionable recommendations that can lead to more profitable outcomes for insurance companies. Such findings helped the team solidify and summarize the recommendations which were categorized into immediate implementations and future work. For immediate implementation, the team suggested revising primary care coverage and premiums, adjusting reimbursement for `Nurse Practitioners` and `Physician Assistants`, standardizing service costs across different geographic locations, and incentivizing non-participating providers to capture more market share.

Future work recommendations included introducing specialty-specific premiums, revising prescription drug coverage, developing a risk stratification model for beneficiaries, investing in data infrastructure, and implementing targeted health promotion programs.

While the analysis provides valuable insights, it is important to acknowledge the assumptions and limitations. These include the assumption of data completeness, accuracy, and consistent categorization. However, limitations such as temporal constraints, the absence of clinical data and detailed provider information, and geographical limitations should be considered. Addressing these limitations in future studies can further enhance the understanding of healthcare utilization and costs within the Medicare system.


<h1 style="color:#0d2599; background-color:#e0eaf8; border: 1px solid #ffffff; padding: 10px 0;">TABLE OF CONTENTS</h1>

[EXECUTIVE SUMMARY](#ExecSum) <br> 
[LIBRARIES & FUNCTIONS](#Libraries) <br> 
[1. PROBLEM STATEMENT](#Problem) <br> 
[2. MOTIVATION](#Motivation) <br> 
[3. METHODOLOGY](#Methodology) <br> 
[4. DATA COLLECTION](#Collection) <br> 
- [4.1 Data Source](#Source) 
- [4.2 Data Description](#Description) 
- [4.3 Data Collection](#Collect) <br> 

[5. DATA PREPROCESSING AND EDA](#Prepare) <br> 
[6. DATA ANALYSIS & VISUALIZATION](#Analytics) <br> 
[7. RESULTS AND DISCUSSION](#Results) <br>
[8. CONCLUSION](#Conclusion) <br> 
[9. RECOMMENDATION](#Recommendation) <br> 
- [9.1 Limitations of the Study](#Limitations)
- [9.2 Recommendation](#Recommendation2)
- [9.3 Future Work](#FutureWork) <br> 

[REFERENCES](#References) <br> 


<h2>ACKNOWLEDGEMENT</h2>

I completed this project with my Sub Learning Teammate <b>Erika G. Lacson</b>.

