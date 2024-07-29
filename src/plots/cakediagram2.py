import matplotlib.pyplot as plt

labels = ['Sinus Rhythm', 'Atrial Fibrilliation', 'Atrial Flutter', 'Premature Atrial Contraction', 'Premature Ventricular Contraction']
sizes = [28971, 5255, 8374, 3041, 1279]
colors = ['tomato', 'darkorange', 'gold', 'mediumseagreen', 'royalblue']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
wedges1, texts1, autotexts1 = ax1.pie(sizes, labels=sizes, colors=colors, autopct='%1.1f%%', shadow=False, startangle=180)
ax1.axis('equal')
ax1.set_title('Physionet 2021\n')
sizes = [7074, 2119, 665, 569, 160]
wedges2, texts2, autotexts2 = ax2.pie(sizes, labels=sizes, colors=colors, autopct='%1.1f%%', shadow=False, startangle=180)
ax2.axis('equal')
ax2.set_title('MyDiagnostick\n')
fig.legend(wedges1, labels, title="Arrhythmias", loc="upper center", bbox_to_anchor=(0.5, 1), ncol=5)
plt.subplots_adjust(top=0.80)
plt.show()
