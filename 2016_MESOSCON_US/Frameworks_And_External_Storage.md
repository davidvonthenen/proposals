### Enabling External Persistent Storage for Frameworks

**Proposal for MesosCon US 2016**  

**Choose a Proposal Title. A proposal must have a short, specific presentation title (containing no abbreviations) that indicates the nature of the presentation.**:  
Enabling External Persistent Storage for Frameworks

**Choose a submission type (Presentation, Panel, Keynote, Lightning Talk)**:  
Presentation

**Select a Speaker for your Proposal. The system will default to the submitter as the author, but can be edited in the submission process. Name, email, title, company, and bio will be required for submission.**:  
David vonThenen  
david.vonthenen@emc.com  
Developer Advocate  

David vonThenen is a Developer Advocate at EMC. He is currently a member of the EMC {code} team which lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos to storage orchestation platforms. Prior to joining EMC {code}, David was a technical architect and development lead for a Backup/Recovery solution at EMC with heavy focus in the virtualization space, VMware in particular. David has previous experience in a wide variety of areas ranging from semiconductors, mainframe computing, and iSCSI/FC/FCoE storage initiators/target.

**Choose the category for your proposal (Developer, Operations, Business/Legal, Wildcard):**  
Developer

**Choose the audience type (Operators, New Users, Enterprise, Contributors):**  
Contributors

**Select the experience level (Beginner, Intermediate, Advanced, Any):**  
Advanced

**Provide us with an abstract about what you will be presenting at the event (900 characters maximum).:**  
Since being unveiled at MesosCon Europe in 2015, external persistent volumes have quickly received a lot of attention with its potential applications. We will discuss using volume drivers as means of provisioning external storage for Mesos Frameworks. We will specifically use the Elastic Search framework (https://github.com/mesos/elasticsearch) as a reference implementation to discuss how the solution of adding external volumes was architected. A demo of the Elastic Search framework will be provided with discussions about how external storage can benefit Frameworks and the applications they manage.

Proposal Outline:
- What is a Framework?
  - Schedulers
  - Executors
- External Storage Considerations for Frameworks
  - Storage is a externally managed.
  - Only consider CPU and memory for Offers
  - Need a concept of a "Node ID". Describe why this Node ID is important.
- "Node ID"
  - Need a mechanism in order to tie a Node to Persistent Storage
  - Needs to support Mesos Agent failover
  - How do we accomplish this?
- Demo Description
  - Supports Docker Volume Driver Interface (DVDI)
  - Support mesos-module-dvdi for non-Docker Executors
- Demo
  - Deploy ElasticSearch Framework using External Storage
    - Discuss the behaviors of the Framework

**Tell us how the content of your presentation will help better the ecosystem (900 characters maximum).:**  
This discussion will help developers understand some of the challenges and solutions that were encountered when enabling external storage for Frameworks, provide insight for potential enhancement requests for Mesos, and expand the Mesos user base understanding of Mesos Frameworks.

**List any technical requirements that you have for your presentation over and above the standard projector, screen and wireless Internet.:**  
None, all demos will be done over AWS.
