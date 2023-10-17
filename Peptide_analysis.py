


Peptide
/Peptide_analysis

from modlamp.analysis import GlobalAnalysis
# Save generated sequences to a .txt file
with open("generated_sequences.txt", "w") as file:
    for sequence in generated_sequences:
        file.write(sequence + "\n")

# Read sequences from the .txt file
with open("generated_sequences.txt", "r") as file:
    sequences = file.readlines()
    sequences = [sequence.strip() for sequence in sequences]
# Perform analysis
ga = GlobalAnalysis(sequences)
ga.all()
# save calculated descriptor to a .csv file
ga.save_descriptor('location/of/your/outputfile.csv', delimiter=',')
import pandas as pd
import matplotlib.pyplot as plt

# Read the .csv file into a pandas DataFrame
df = pd.read_csv('outputfile.csv')

# Plot the data
plt.plot(df['column_name'])
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Plot Title')
plt.show()
