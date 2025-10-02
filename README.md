# VESC ML Telemetry Analyzer
*(VMLTA, it flows off the tongue)*

### Summary

This is my capstone project for the WGU B.S. in Computer Science. The purpose of the project is to train a machine learning model to identify
rider behaviors for onewheel style PEVs using telemetry data from the VESC open-source electronic speed controller software. I have developed
a data preprocessing pipeline for ingestion, formatting, cleaning, normalizing sample rates, and classifying behaviors on a confidence scale 
through a combination of Python scripts and manual segment labeling in Label Studio. The motivation for the project comes from my hobbyist
obsession with VESC onewheels, and the possibility that training a machine learning model on VESC telemetry could have a variety of useful
applications for the community.

The most plausible use case of machine learning on telemetry data is for troubleshooting and configuration error detection. This initial proof 
of concept does not include training for troubleshooting or configuration purposes, it is trained to identify primitive rider behaviors, such 
as acceleration, braking, cruising, and turning. Training a model to detect improper configurations or faulty hardware would require, among 
other things, a larger dataset gathered from misconfigured devices, and a diverse array of devices in order to make improper configuration 
behaviors identifiable in a way that would be useful for other riders. This pipeline and training model provides a useful starting point from 
which to build out the features of a more useful model that is capable of providing troubleshooting assistance through recognition of telemetry 
patterns that result from improperly configured devices. If there is enough community interest in this system, it would be exciting to explore 
further potential to extend the model features beyond troubleshooting into configuration optimizations through event, behavior, and tune analysis 
to provide recommendations for device tuning adjustments.

### Preprocessing Pipeline Overview

The preprocessing pipeline scripts accept a raw VESC telemetry log file (CSV) as input, and then reformat the data from semi-colon delineated
lists into a structured table format with columns for each data channel and rows for each sequential data sample. Next, the pipeline normalizes
the sample rate to a 100ms (10hz) sample rate by adding rows/samples at 100ms increments from the start time and interpolating values for each
channel to fill these new rows. The interpolation was necessary because the telemetry logs have an inconsistent sample/communication rate over
BLE from the device to the mobile application that varies from 5ms - 250ms. Sample rate normalization and interpolation also serve to smooth 
the data points through interpolation in a way that resembles a rolling average method. For the training portion of the pipeline, to include 
behavior classifications in the training data, the script inserts new columns for timestamps converted from the VESC default UTC time since 
midnight in ms to PST for easier alignment with the video footage that will be used for labeling data segments. The script also adds the columns 
for the behavior classifications themselves, which are expressed in confidence values on a scale of 0.0 to 1.0. All of the behavior classification 
values are initialized to NaN, as the training will be conducted in a semi-supervised fashion, with only clear behaviors being explicitly labeled 
for training.
 
After the scripts have reformatted the data, added additional timestamps, other potentially useful identification labels, and additional behavior 
classification columns, it is exported as a CSV that is ready for labeling in Label Studio. I have included the XML file that I used for the Label
Studio project to set up behavior classifications on a confidence scale. This portion of the project involved synchronizing the action cam footage,
screen recordings of the VESC Tool and Float Package APP UI, and log data in order to manually label the behavior segments of the log files in the 
Label Studio by referencing the corresponding videos and narration. Another CSV can then be exported from Label Studio with labeled time intervals 
for each labeled behavior. This CSV is processed through another Python script that merges the labels from the LS CSV with the processed CSV ride 
log to apply the label segments to the log data. This script also applies exclusion rules for behaviors that logically cannot occur simultaenously 
(e.g. left turn + right turn, acceleration + braking) to simplify the labeling process. For example, if a data segment is labeled as a left turn 
with a confidence of 0.5, the script writes in that value over the specified time segment, as well as confidence values of 0.0 for right turn and 
carve right. At this point, the CSV and parquet files produced from the script are ready for the machine learning model.

### Model Training