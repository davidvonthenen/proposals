**Proposal for ApacheCon US 2017**  

**Submission Type (BoF, Lightning Talk, Panel Discussion, Presentation, Tutorial, Mini Summit, Lab)**  
Presentation

**Category (Developer, Operations, Business, Wildcard):**  
Developer

**Biography. Provide a biography that includes your employer (if any), ongoing projects and your previous speaking experience.:**  
David vonThenen  
david.vonthenen@emc.com  

David vonThenen is an Open Source Engineer at {code} by Dell EMC. The {code} team lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos, Docker, Kubernetes, and storage orchestration platforms. Prior to joining {code} by Dell EMC, David was a technical architect and development lead for a Backup/Recovery solution with a heavy focus in the virtualization space, VMware in particular.

**Abstract Title**  
How Container Schedulers and Software-based Storage will Change the Cloud

**Abstract. Provide an abstract that briefly summarizes your proposal. This is the abstract that will be posted on the website schedule, so please ensure that it is in complete sentences (and not just bullet points) and that it is written in the third person (use your name instead of I).:**  
Persistent applications can be complex to manage and operate at scale but tend to be perfect for modern schedulers like Apache Mesos. The current trend in Containers is taking their ephemeral nature and turning it upside-down by running databases, key/value stores, WordPress, and etc within them. Internal direct attached storage and external storage are both options in running your long-running, persistent applications. The problem is how do you run these applications and containers in production environments?

This talk outlines how 2 Layer Scheduling, as known as the Offer-Accept model, found in Mesos and Software-based Storage enables deployment of managed frameworks and tasks while maintaining high availability, scale-out growth, and automation. This combination of technology will help build a "Skynet" like architecture for persistent applications and containers in the cloud.

**Audience. Describe who the audience is and what you expect them to gain from your presentation.:**  
The audience will learn about a critical principle concepts in container schedulers. Many container schedulers already have 2 Layer Scheduling present in their Ecosystem while others have the feature as priority number one in their roadmaps. 2 Layer Scheduling will fundamentally change the way we consume container runtimes, like Docker, rkt, and etc.

**Experience Level. (Beginner, Intermediate, Advanced, Any):**  
Intermediate

**Benefits to the Ecosystem. Tell us how the content of your presentation will help better the ecosystem. This could be for Linux, open source, open cloud, embedded, etc.:**  
This presentation will discuss the concept of Software-based Storage and how its deployment into cloud environments like Amazon AWS and Google Compute Engine (GCE) will compliment container schedulers. Those not familiar with Software-based Storage will also be given a high-level introduction to the benefits of the technology.

**Technical Requirements:**  
None, all demos will be done over AWS.
