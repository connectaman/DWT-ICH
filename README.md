# DWT-ICH
Intracranial haemorrhage (ICH) segmentation on Brain CT image using DWT

Head injury is a chief cause for morbidity and mortality worldwide and Traumatic head
injuries signify the chief cause of neurological disability. Head injuries may range from a
simple bump on the head to a skull fracture and may cause brain damage and even death.
A traumatic brain injury (TBI) occurs when the brain is damaged, usually because of an
accident (or any trauma) and can damage the brain, causing a blood clot or bruising in the brain
identified as an Intracranial Hematoma (ICH). ICH records serious consequence of head injury.
ICH are the rupture of a blood vessel which leads to the collection of blood in brain tissues or
empty spaces which can be life-threatening. The most common cause of ICH normally reported
in our country are road traffic accidents (RTA) followed by falls and assaults. India is a
populous country with over a billion people and there is approximately 1 radiologist for every
100,000 population with a majority of them in the urban setup, Indian rural population of more
than 70% is deprived of these doctors. The unavailability of these specialists is a grave concern
to the well-being of the health care to the nation. The mainstay in the diagnosis of an ICH is
the Computed Tomography (CT) scan of the head which is the definitive tool for accurate
diagnosis of an ICH following trauma and gives an objective assessment of structural damage
to brain.


### Working 

![flowchat](https://github.com/connectaman/DWT-ICH/blob/main/images/flowchart.PNG)


-----------------------------------------------------------

### Usage

```python
# import the dwt_ich library
from  dwt_ich impor ICH
import matplotlib.pyplot as plt
# path of brain image
img_p =  r'path to brain ct image'  
#create object of ICH class
brain = ICH()
# call the predict method and pass the image path
res = brain.predict(img_p)
# in return you get back image after segmentation
plt.imshow(res,cmap='gray')
# Display the resultant image
plt.show()
```


-----------------------------------------------------------


### Result

![result](https://github.com/connectaman/DWT-ICH/blob/main/images/result.PNG)
![result](https://github.com/connectaman/DWT-ICH/blob/main/images/result2.PNG)
![result](https://github.com/connectaman/DWT-ICH/blob/main/images/result3.PNG)

