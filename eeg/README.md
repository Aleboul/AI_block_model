# Study of CHB-MIT Scalp EEG database.

## Data Description

This database, collected at the Children’s Hospital Boston, consists of EEG recordings from pediatric subjects with intractable seizures. Subjects were monitored for up to several days following withdrawal of anti-seizure medication in order to characterize their seizures and assess their candidacy for surgical intervention.

Recordings, grouped into 23 cases, were collected from 22 subjects (5 males, ages 3–22; and 17 females, ages 1.5–19). (Case chb21 was obtained 1.5 years after case chb01, from the same female subject.) The file SUBJECT-INFO contains the gender and age of each subject. (Case chb24 was added to this collection in December 2010, and is not currently included in SUBJECT-INFO.) More information can be found in the corresponding website: https://physionet.org/content/chbmit/1.0.0/

Each case (chb01, chb02, etc.) contains between 9 and 42 continuous .edf files from a single subject. Hardware limitations resulted in gaps between consecutively-numbered .edf files, during which the signals were not recorded; in most cases, the gaps are 10 seconds or less, but occasionally there are much longer gaps. In order to protect the privacy of the subjects, all protected health information (PHI) in the original .edf files has been replaced with surrogate information in the files provided here. Dates in the original .edf files have been replaced by surrogate dates, but the time relationships between the individual files belonging to each case have been preserved. In most cases, the .edf files contain exactly one hour of digitized EEG signals, although those belonging to case chb10 are two hours long, and those belonging to cases chb04, chb06, chb07, chb09, and chb23 are four hours long; occasionally, files in which seizures are recorded are shorter.


All signals were sampled at 256 samples per second with 16-bit resolution. Most files contain 23 EEG signals (24 or 26 in a few cases). The International 10-20 system of EEG electrode positions and nomenclature was used for these recordings. In a few records, other signals are also recorded, such as an ECG signal in the last 36 files belonging to case chb04 and a vagal nerve stimulus (VNS) signal in the last 18 files belonging to case chb09. In some cases, up to 5 “dummy” signals (named "-") were interspersed among the EEG signals to obtain an easy-to-read display format; these dummy signals can be ignored.

## File Description

* `script_seizures.py`, generate a csv file containing the block maxima (here, block of 4 minutes) of the time series without seizures. This file contains 140293 seconds of records.
* `script_seizures.py`, generate a csv file containing the block maxima (here, block of 5 seconds) of the time series with seizures. This file contains 558 seconds of records.
* `analysis_seizures.py`, performs clustering according to an AI block model of the CHB-MIT Scalp EEG database for patient 05. The analysis is performed where seizures are observed.
* `analysis_wo_seizures.py`, Performs clustering according to an AI block model of the CHB-MIT Scalp EEG database for patient 05. The analysis is performed where no seizures are observed.
