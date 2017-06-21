**Proposal for MesosCon US 2017**  

**Submission Type (BoF, Lightning Talk, Panel Discussion, Presentation, Tutorial, Mini Summit, Lab)**  
Presentation

**Category (Developer, Operations, Business, Wildcard):**  
Operations

**Biography. Provide a biography that includes your employer (if any), ongoing projects and your previous speaking experience.:**  
David vonThenen  
david.vonthenen@dell.com  

David vonThenen is an Open Source Engineer at {code} by Dell EMC. The {code} team lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos, Docker, Kubernetes, and storage orchestration platforms. Prior to joining {code} by Dell EMC, David was a technical architect and development lead for a Backup/Recovery solution with a heavy focus in the virtualization space, VMware in particular.

**Abstract Title**  
Robust Applications in Mesos using External Storage

**Abstract. Provide an abstract that briefly summarizes your proposal. This is the abstract that will be posted on the website schedule, so please ensure that it is in complete sentences (and not just bullet points) and that it is written in the third person (use your name instead of I).:**  
Containers are starting to reach the masses and people are using them in ways other than what was originally intended. We now find persistent applications like SQL and NoSQL databases being run in container schedulers like Mesos, but how to we guarantee data availability for production applications in the wake of compute node failures? There are options for using direct attached or external storage, but the devils in the details, as choices in storage types have significant repercussions.

We will discuss the benefits and challenges of using direct attached or external storage and how that impacts applications running in production environments. The trade-offs of each decision have interesting consequences starting from initial deployment to "day 2" operations and even how these applications tolerate system failures.

**Audience. Describe who the audience is and what you expect them to gain from your presentation.:**  
The audience will learn about different persistence models in Mesos and how they affect modern applications in production. We will highlight some projects in the community as well as highlight recent work embedded in Mesos 1.0+ that enable external storage for its end-users. A demo will be presented showing how a NoSQL database can take advantage of these types of persistence models.

**Experience Level. (Beginner, Intermediate, Advanced, Any):**  
Intermediate

**Benefits to the Ecosystem. Tell us how the content of your presentation will help better the ecosystem. This could be for Linux, open source, open cloud, embedded, etc.:**  
This presentation will discuss how users can deploy applications for production Mesos environments, storage orchestration engines to provision persistent volumes, and features present in Mesos that can be used today. In the demo, we will briefly highlight the capabilities that NoSQL databases provide and how Mesos works seamlessly in cloud environments like AWS.

**Technical Requirements:**  
None, all demos will be done over AWS.
