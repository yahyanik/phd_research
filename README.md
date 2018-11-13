all files are written for my project on detecting and tracking humans on cctv in realtime.
# phd_research
In this folder the main file is the app.py that will call for the rest. The detection is based on the Haar Cascaded and the KCF is for tracking a human as an object in a automatic way.

After detection the coordintion of the person are given to the tracker in a queue and if the object is not active, not moving, the tracker will delete the object from the queue.

The detection and tracking architectures can be changed based on the lines that are used.
