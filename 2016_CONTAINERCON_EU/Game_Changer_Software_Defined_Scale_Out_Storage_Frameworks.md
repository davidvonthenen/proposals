**Proposal for ContainerCon EU 2016**  

**Submission Type (BoF, Lightning Talk, Panel Discussion, Presentation, Tutorial, Mini Summit, Lab)**  
Presentation

**Category (Developer, Operations, Business, Wildcard):**  
Operations

**Biography. Provide a biography that includes your employer (if any), ongoing projects and your previous speaking experience.:**  
David vonThenen  
david.vonthenen@emc.com  
Developer Advocate  

David vonThenen is a Developer Advocate at EMC {code}. The {code} team lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos to storage orchestration platforms. Prior to joining EMC {code}, David was a technical architect and development lead for a Backup/Recovery solution with heavy focus in the virtualization space, VMware in particular. David has previous experience in areas ranging in semiconductors, mainframe computing, and iSCSI/FC/FCoE storage initiators and targets.

**Abstract Title**  
Game Changer: Software Defined Storage and Container Schedulers

**Abstract. Provide an abstract that briefly summarizes your proposal. This is the abstract that will be posted on the website schedule, so please ensure that it is in complete sentences (and not just bullet points) and that it is written in the third person (use your name instead of I).:**  
One problem of running Enterprise Applications in container schedulers, like Apache Mesos and Kubernetes, has been making applications and their data highly available. To date, utilizing local disks on compute nodes has given us data persistence, but unfortunately does solve the data mobility problem required to make applications tolerate Agent node failures.

We will discuss what Software Defined Storage (SDS) is, how Software Defined Storage can transform local storage into an external globally accessible pool, how Mesos clusters can overcome this data mobility problem, and more importantly do so in such a way that is simple and easy to consume using an Apache Mesos Framework as a reference model. Will have a demonstration of Mesos Framework that will deploy a scale out software defined storage platform and deploy applications leveraging this new type of storage.

**Audience. Describe who the audience is and what you expect them to gain from your presentation.:**  
This session is geared toward people looking to understand more about container schedulers and be introduced to software defined storage platforms. We will take a look at what is require to run applications on schedulers and container runtimes in highly available production environment.

**Experience Level. (Beginner, Intermediate, Advanced, Any):**  
Intermediate

**Benefits to the Ecosystem. Tell us how the content of your presentation will help better the ecosystem. This could be for Linux, open source, open cloud, embedded, etc.:**  
This session will introduce the concepts of Software Defined Storage to users who might not be familiar with the subject, give guidance in running Enterprise Applications in Mesos clusters, and review Mesos Frameworks which is driving the enablement of storage platform.

**Technical Requirements:**  
None, all demos will be done over AWS.
