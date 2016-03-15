### Frameworks and Two Layer Scheduling for Docker

**Proposal for DockerCon US 2016**

**First Name:**  
David

**Last Name:**  
vonThenen

**Email Address:**  
david.vonthenen {at} emc {dot} com

**Title:**  
Frameworks and Two Layer Scheduling for Docker

**Company:**  
EMC

**Personal Blog:**  
http://dvonthenen.com

**List the previous industry conferences you've spoken at?:**  
EMC World

**Link to videos from previous talks:**  
https://www.youtube.com/watch?v=YE2bLuP5M2c

**Twitter Handle**  
dvonthenen

**Speaker bio (75 words max):**  
David vonThenen is a Developer Advocate at EMC. He is currently a member of the EMC {code} team which lives and breathes Open Source by making contributions to the community in a wide variety of projects ranging from Apache Mesos to storage orchestration platforms. Prior to joining EMC {code}, David was a technical architect and development lead for a Backup/Recovery solution with heavy focus in the virtualization space, VMware in particular. David has previous experience in areas ranging in semiconductors, mainframe computing, and iSCSI/FC/FCoE storage initiators and targets.

**What's your relationship to Docker ?:**  
Contributor

**Session Title:**  
Frameworks and Two Layer Scheduling for Docker

**Abstract:**  
If applications all behaved in the same way, we would only have one solution to scheduling resources. In the real world, we know that is never the case. Two layer scheduling provides a mechanism whereby you can specialize your application workloads to address a wide variety of use cases.

In this session, we will take a look at how Apache Mesos achieves two layer scheduling via Frameworks by briefly reviewing what Apache Mesos is, discuss the components that make up a Framework, and discover how we customize the allocation of resource. Then we will examine how Docker containers in concert with a Docker Volume Driver can quickly enable provisioning an application that is highly available with very little effort. Finally, we will see Frameworks in action as we demonstrate deploying a Framework in order to highlight their capabilities and how you can rapidly deploy and configure applications.

**Outline:**  
Proposal Outline:
- Quick Review of Mesos
  - Describe Mesos
    - Singly managed resource pool
    - Scheduler/Dispatcher of Applications
  - Mesos
    - Master Nodes (Zookeeper Quorum)
    - Agent (Slave) Nodes
  - Application Support
    - Generic
    - Framework
  - Framework Review
    - Scheduler
    - Executor
  - Application Examples
  - Framework Examples
- What is a Framework?
  - Schedulers
  - Executors
- Framework Architecture
  - What are the moving parts?
  - Who is talking to who? (Interaction Diagram)
- What does a Framework buy me?
  - Two Layer Scheduling
  - Use cases from other popular frameworks
    - GPU
    - Disperse application deployment
    - Etc
- What Docker bring to the table
  - Docker Hub
    - Ease of Deployment
    - Configuration
    - Revision Management
  - Docker Volume Drivers
    - What is a Docker Volume Driver?
    - Persistent Storage
    - High Availability
- Demo
  - Describe Demo Configuration
    - Elastic Search Mesos Framework
    - 3 Node Elastic Search Cluster
    - Start Deploy
  - Disaster!
    - Node Failure
    - Disk Failure
    - Downtime (or performance hit) to RAID rebuild
    - Network failure or partitioning event!
  - Answer: External Storage!
    - Node recovers from where it left off (quicker recovery)
    - No node rebuild (hardware rebuild fast, software slow)

**What are the key takeaways from your session? :**  
- Better understanding of of two layer scheduling
- Introduction to Docker Volume Drivers
- How persistent storage enables more use cases for Docker containers
- Show cases other projects in the open source community
- All components used for the demo is available on GitHub

**Are you able to share details about your application architecture/design and implementation results like metrics?:**  
Yes

**Keywords**  
Docker Volume Driver, High Availability, Scheduler

**Can your CFP be turned into a lightning talk?:**  
No

**What theme do you believe your talk fits into?:**  
Use Case

**Expertise level:**  
Advanced

**Who's the main target audience?:**  
SysAdmin, Ops Engineer, Site Reliability Engineer

**Is your talk related to any of the following Docker Projects?:**  
Docker Engine
Docker Hub
Other: Docker Volume Driver

**Does your presentation have the participation of a woman, person of color, person with disabilities, or member of another group often underrepresented at tech conferences?:**  
No

**Will you have a co-presenter?:**  
No

**Agree to the code of conduct:**  
Yes
