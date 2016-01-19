### Docker, Mesos, OpenStack, and Storage Persistence

**Proposal for OpenStack Summit US 2016**

**Please select the level of your presentation content**  
Intermediate

**YouTube and other services limit the length of your presentation's description. We will take the first 100 characters of your abstract to display in the YouTube description. Abstract* (max 1000 chars)**  
This tutorial will take an existing production Apache Mesos cluster on top of OpenStack and walk-through adding storage persistence via Docker volume driver isolator and then will implement some application use cases leveraging this persistent storage. We will also discuss why this implementation insulates and protects your workloads from the underlying management infrastructure such as but not limited to OpenStack, VMware, AWS, and etc.

**What is the problem or use case you’re addressing in this session?* (max 1000 chars) Characters left: 1000**
This tutorial will take an existing production Apache Mesos cluster on top of OpenStack and walk-through adding storage persistence via an external storage orchestration engine called REX-Ray (https://github.com/emccode/rexray), and a Mesos Docker volume driver isolator called mesos-module-dvdi (https://github.com/emccode/mesos-module-dvdi). Then we will learn how we can leverage the storage persistence in this Mesos Cluster by implementing simple and complex uses cases that have been enabled via external storage.

For those not familiar with Apache Mesos, we will talk about what Mesos provides us in terms of capabilities and why its called the Data Center Operating System (DCOS). Then we will kick the tires by deploying some basic applications using Apache Marathon UI and REST API to get familiar with its interfaces.

We will then enhance this Mesos Cluster's capabilities by adding storage persistence. From there we will implement various use cases and discuss the ramifications of these application configurations. We will also discuss why this implementation insulates and protects your workloads from the underlying management infrastructure such as but not limited to OpenStack, VMware, AWS, and etc.

**What should attendees expect to learn?* (max 1000 chars)**  
Adding external persistent storage to your Apache Mesos cluster will fundamentally change the way in which you use Mesos.

**Why should this session be selected?* (max 1000 chars)**  
This is a different take on OpenStack and how you should consume OpenStack resources. Should OpenStack go defunct, this will give you a mechanism to leave the OpenStack infrastructure behind.

**What is the general topic of the presentation?**  
Architectural Decisions
