# Public Data Project - Flood Warning Classification

# Executive Summary

As part of Module 5 of my BSc Data Science Apprenticeship, which focused on applying public datasets to real-world data science problems, I developed a flood warning predictor for my local flood warning area. The aim of this project was to gain a deeper understanding of how the Environment Agency issues flood warnings, and to explore whether these warnings are based on specific thresholds in river level, flow, or rainfall. I was particularly interested in discovering whether warnings are automated, what levels are indicative of potential flooding, and which time-based features—such as lagged readings—can help predict flood events.

The project exclusively utilised data from the Environment Agency Hydrology API, accessed via the DEFRA Data Services Platform. I selected Flood Warning Area 761 as a prototype, allowing for focused model development and experimentation.

To enhance the predictive power of the model, I engineered a variety of time-based features. These included lagged river levels (1h and 2h), short-term and medium-term rolling averages, and exponentially weighted moving averages (EWMAs) over time windows ranging from 3 to 24 hours. Rainfall was incorporated as cumulative totals over the past 12 hours, 3 days, and 2 weeks to account for both short-term and long-term saturation effects. Flow-based features were also included, mirroring the lag, average, and EWMA structure used for river levels.

The final model, a Random Forest classifier, demonstrated strong performance with a precision of 0.88, recall of 0.76, and an F1 score of 0.82. This model successfully identified most hours associated with flood warnings while maintaining a low false-alarm rate, outperforming earlier versions significantly. In addition, a Decision Tree model was trained on the top features to extract interpretable thresholds, offering a clear set of rules for future monitoring and potential integration into operational decision-making.

This project demonstrates the value of open hydrological data and machine learning in developing localised flood prediction systems. The results suggest that meaningful early warnings can be generated using only public data, and that feature-rich models can improve both accuracy and interpretability in critical environmental applications.
