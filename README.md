# Evaluating Hierarchical Models Using a WGAN

## Stop and Frisk paper
[Access Gelman's Stop and Frisk paper here](http://www.stat.columbia.edu/~gelman/research/published/frisk9.pdf)
The relevant models (1, 3, and 4) are on pages 5-6.

## Exploratory Analysis with Data Pre Processing.ipynb

This jupyter notebook contains a combination of the exploratory analysis as well
as necessary data processing for the behavioral learning experiment and the
stop and frisk data.

For the stop and frisk data, I read in the stop reports from the Stop, Question,
and Frisk database which is stored in the sqf*.csv files. I use sqf-2014.csv
to calculate the number of arrests in the previous year for a given ethnicity
in a given precinct. I use a 15 month period between Jan sqf-2015.csv and
Mar sqf-2016.csv. My study focuses on fitting a hierarchical model only for
weapons related crimes, and I filter down the stop reports accordingly for this.
I also determine the 3 categories of ethnic composition (<10% black, 10-40% black,
and >40% black) using the information in NYC_Precinct_census_2017.csv.

For arrests in 2014, I encountered precincts which did not have arrests
for one or more of the considered ethnicities. I handled this in two different
ways based on the hierarchical models that I later fit. In 2014_arrests.csv,
precincts that did not have arrests in 2014 have a value of 1e-6, and these values
were used in models that directly calculated the log of this arrests in 2014 value. In
2014_arrest_zeros.csv, precincts that did not have arrests in 2014 have a value
of 0, and these values were used in models that required integer values for
arrests in the previous year. Both of these previous .csv files contain the number
of arrests for only weapons related crimes. The corresponding files ending with
all_crimes.csv handle missing values in the same manner as their respective counterparts
except these files calculate the number of arrests for all suspected crimes rather
than just weapons related crimes. Finally, 20152016.csv contains the number of stops
in the 15 month period from 2015-2016.

### HMM_BE_SF.R
This R file contains all fitted hierarchical models that I used in my project.

First, you will see the hierarchical model for the complex synthetic
distribution which is stored in synDist.csv.

Next, you will see two models for the behavioral learning experiment with dogs.
The code for these STAN models is taken from the Gelman data analysis textbook
and online resources.

Finally, you will see three models for the stop and frisk data. Before I fit
the models, I split the number of arrests in the previous year and the number of
stops in the fifteen month period for each precinct into the 3 categories
based on ethnic composition.    
