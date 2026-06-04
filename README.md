# ai-burnout-project
python scripts for my Medium Article on AI-induced Burnout: [Obtaining AI-induced Burnout Triggers using Gradient Boosting Trees and SHAP](https://medium.com/@mrobith95/obtaining-ai-induced-burnout-triggers-using-gradient-boosting-trees-and-shap-7f1c6c424489)

## Background
Nowadays, generative AI (genAI) is increasingly being applied in various business sectors. According to [McKinsey's 2025 State of AI survey](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-how-organizations-are-rewiring-to-capture-value), 65% of organizations used genAI in at least 1 business function in March 2024. This percentage almost doubled from last year. This rapid adoption of genAI technology will not be free from side effects, such as AI-induced burnout experienced by employees. [ZDNet's report on Upwork's research](https://www.zdnet.com/article/heavy-ai-use-at-work-has-a-surprising-relationship-to-burnout-new-study-finds/) says that freelancers who use AI heavily are 88% more likely to experience burnout. It is well known that burnout might cause [absenteeism, reduced professionalism,](https://www.ncbi.nlm.nih.gov/books/NBK614516/) even [hurt company's revenue](https://hbr.org/2025/06/employee-stress-is-a-business-risk-not-an-hr-problem). Thus, understanding what factors lead to AI burnout accurately can help companies take strategic steps to address the problem.

This project consists of 2 steps to answer this question. The first step to fit a machine learning model, namely gradient boosting regressor, to predict AI burnout level given several variables. The second step is to use SHAP (SHapley Additive exPlanations) to extract information regarding which variables actually drive that prediction.

## Quick Notes
* Dataset: [AI Worker Burnout & Attrition Risk Dataset](https://www.kaggle.com/datasets/nudratabbas/ai-worker-burnout-and-attrition-risk-dataset)
* Model: Gradient Boosting (main modelling), SHAP (explainer)
* Metric: Mean Absoulte Error
* Benchmark: Median of target
* Packages used: Available on requirements.txt

## Intended Pipeline
`download_data.py` ➡️ `data_prep.py` ➡️ `data_preprocessing.py` ➡️ `feature_eng.py` ➡️ `modelling.py` ➡️ `predict.py` ➡️ `shap_explain.py`
