BASE_SYSTEM_PROMPT = "You are a helpful assistant."

ANOMALY_DICT = {
  "Normal Sequence": (0, "There are no abnormal situations in this time series."),
  "Point Anomaly": (1, "A single data point significantly deviates from the local or global pattern of the sequence."),
  "Periodic Change Anomaly": (2, "The original periodic pattern in the time series is disrupted, such as the period being broken or the amplitude becoming anomalous."),
  "Trend Change Anomaly": (3, "A sudden change in the long-term trend of the time series."),
  "Change Point Anomaly": (4, "Statistical properties (e.g., mean, variance) of the sequence change abruptly at certain points."),
  "Distributional Change Anomaly": (5, "The statistical distribution of the time series changes significantly."),
  "Amplitude Anomaly": (6, "The amplitude of data points exceeds the normal upper and lower bounds."),
  "Pattern Change Anomaly": (7, "The pattern of the time series suddenly changes from one form to another."),
  "Sparse Anomaly": (8, "Isolated anomalous patterns occasionally appear in a long time series."),
  "Repeated Value Anomaly": (9, "Continuous or intermittent repeated values disrupt the normal fluctuation pattern."),
  "Sudden Flatline Anomaly": (10, "The time series suddenly becomes a flat line with no normal fluctuations."),
  "Drift Anomaly": (11, "The data of the time series gradually drifts away from the normal level."),
  "Sudden Spike Anomaly": (12, "The data suddenly spikes or drops within a short time and then returns to normal."),
  "Continuous Segment Anomaly": (13, "A continuous segment of data points deviates from the normal pattern."),
  "Nonlinear Pattern Anomaly": (14, "Nonlinear changes appear in the sequence, breaking the original linear rule."),
}

USER_DETECTION_PROMPT = """You are an expert in time series anomaly detection. We provide a time series (called Observation), you should give us the anomaly type (called Action) and its reasons (called Thought). 
Thought steps can infer the current abnormal situation of time series.
Action is an abnormal category with the following 0-14 types, where 0 is a normal category. The explanation of 0-14 actions are as follows:

0. Normal Sequence: There are no abnormal situations in this time series.
1. Point Anomaly: A single data point significantly deviates from the local or global pattern of the sequence.  
2. Periodic Change Anomaly: The original periodic pattern in the time series is disrupted, such as the period being broken or the amplitude becoming anomalous.  
3. Trend Change Anomaly: A sudden change in the long-term trend of the time series.  
4. Change Point Anomaly: Statistical properties (e.g., mean, variance) of the sequence change abruptly at certain points.  
5. Distributional Change Anomaly: The statistical distribution of the time series changes significantly.  
6. Amplitude Anomaly: The amplitude of data points exceeds the normal upper and lower bounds.  
7. Pattern Change Anomaly: The pattern of the time series suddenly changes from one form to another.  
8. Sparse Anomaly: Isolated anomalous patterns occasionally appear in a long time series.   
9. Repeated Value Anomaly: Continuous or intermittent repeated values disrupt the normal fluctuation pattern.  
10. Sudden Flatline Anomaly: The time series suddenly becomes a flat line with no normal fluctuations.  
11. Drift Anomaly: The data of the time series gradually drifts away from the normal level.  
12. Sudden Spike Anomaly: The data suddenly spikes or drops within a short time and then returns to normal.   
13. Continuous Segment Anomaly: A continuous segment of data points deviates from the normal pattern.  
14. Nonlinear Pattern Anomaly: Nonlinear changes appear in the sequence, breaking the original linear rule.    

The anomaly detection of each time series is divided into three steps: Observation, Thought, Action. After analyzing each observation, please provide the next Thought and next Action. Here are some examples:

Observation: [1.2, 1.3, 1.1, 1.2, 5.0, 1.3, 1.2, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3]
Thought: A single value (5.0) significantly deviates from the overall range of the surrounding data points, which mostly hover around 1.2, making it an isolated anomaly.
Action: Point Anomaly

Observation: [1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 1.0, 0.5, 3.0, 3.5, 3.0, 2.5, 3.0, 3.5, 3.0, 2.5]
Thought: The periodic pattern of [1.0, 1.5, 1.0, 0.5] is disrupted starting at index 8, where the sequence shifts to a new amplitude pattern of [3.0, 3.5, 3.0, 2.5].
Action: Periodic Change Anomaly

Observation: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]
Thought: The initial trend is a slow linear increase, but starting from index 8, there is a sudden and steeper upward trend, indicating a change in the long-term trajectory.
Action: Trend Change Anomaly

Observation: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
Thought: The mean value shifts abruptly after index 7, with a noticeable jump from around 1.7 to 3.0, altering the statistical properties of the sequence.
Action: Change Point Anomaly

Observation: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
Thought: The statistical distribution changes starting at index 8, with the values increasing at a faster rate compared to the earlier uniform increment.
Action: Distributional Change Anomaly

Observation: [10, 12, 11, 13, 50, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12]
Thought: The value 50 is an extreme outlier compared to the surrounding amplitude, which stays within the range of 10 to 13.
Action: Amplitude Anomaly

Observation: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
Thought: The sequence transitions from a linear increasing pattern to a linear decreasing pattern at index 8, indicating a sudden shift in the overall pattern.
Action: Pattern Change Anomaly

Observation: [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0]
Thought: Isolated high values (10 and 15) appear sporadically in an otherwise flat sequence of zeros.
Action: Sparse Anomaly

Observation: [0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
Thought: Repeated values (0.5 and 0.2) disrupt the normal fluctuation pattern of the sequence.
Action: Repeated Value Anomaly

Observation: [1.0, 1.2, 1.1, 1.3, 1.2, 1.1, 1.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Thought: The sequence transitions abruptly into a flatline of zeros, losing normal fluctuations.
Action: Sudden Flatline Anomaly

Observation: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
Thought: The values gradually drift upward without returning to the original range, deviating from the normal level of fluctuation.
Action: Drift Anomaly

Observation: [0.5, 0.6, 0.7, 0.8, 10.0, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8]
Thought: The value 10.0 spikes suddenly and returns to the normal range immediately after, disrupting the sequence temporarily.
Action: Sudden Spike Anomaly

Observation: [0.5, 0.6, 0.7, 0.8, 3.0, 3.0, 3.0, 3.0, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8]
Thought: A continuous segment of constant values (3.0) from index 4 to 7 interrupts the expected variation in the sequence.
Action: Continuous Segment Anomaly

Observation: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
Thought: The sequence exhibits exponential growth, deviating significantly from the original linear trend.
Action: Nonlinear Pattern Anomaly

Observation: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
Thought: The sequence exhibits a consistent linear increase with no irregularities, sudden changes, or disruptions, following a predictable and smooth trend.
Action: Normal Sequence

Here is a time series observation that we need to check for anomaly categories. The oberservation is from domain of {our_source}.
Please make a Thought judgment and put your final Action within \\boxed1{{}} and \\boxed2{{}} respectively, where action must just be a category name not id.
Observation: {our_observation}
Thought: \\boxed1{{}}
Action: \\boxed2{{}}
"""