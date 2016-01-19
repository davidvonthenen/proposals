### Docker, Mesos, and Storage Persistence

**Proposal for Velocity US 2016**  

**Description:**  
This 3 Hour Tutorial will take an existing production Apache Mesos cluster and walk-through adding storage persistence via Docker volume driver isolator and then will implement some simple and complex application use cases leveraging this persistent storage.

**Topic:**  
Infrastructure Reimagined, Database, Deployment, DevOps, Enterprise

**Session Type:**  
3 Hour Tutorial

**Abstract* (Longer, more detailed description (3-6 paragraph, bullet points welcome) of your presentation to help the program committee understand what you will cover. Please keep in mind that if your proposal is chosen, this abstract will also appear on the website to help conference attendees decide if it's right for them. Note that our copywriters may edit it for consistency and O’Reilly voice.)**  
This 3 Hour Tutorial will take an existing production Apache Mesos cluster and walk-through adding storage persistence via an external storage orchestration engine called REX-Ray (https://github.com/emccode/rexray), and a Mesos Docker volume driver isolator called mesos-module-dvdi (https://github.com/emccode/mesos-module-dvdi). Then we will learn how we can leverage the storage persistence in this Mesos Cluster by implementing simple and complex uses cases that have been enabled via external storage.

For those not familiar with Apache Mesos, we will talk about what Mesos provides us in terms of capabilities and why its called the Data Center Operating System (DCOS). Then we will kick the tires by deploying some basic applications using Apache Marathon UI and REST API to get familiar with its interfaces.

We will then enhance this Mesos Cluster's capabilities by adding storage persistence. From there we will implement various use cases and discuss the ramifications of these application configurations.

1. Apache Mesos
  - What is Apache Mesos?
  - Mesos Architecture
  - What does Mesos buy me?
  - What is Marathon?
2. Hands On Lab: Mesos and Marathon
  - Explore the Marathon UI (end users will navigate on their own with instruction)
    - Deployments tab
    - About (Current master, Marathon info, etc)
    - List of Apps tab
  - Deploy an Application using the Marathon UI
    - Standup Simple Webserver
    - Get application info (what Slave/Agent is it on, etc)
    - Suspend the application
    - Scale up the application
    - Destroy
  - Deploy an Application using the REST API
    - Standup a complex webserver
    - Get application info (what Slave/Agent is it on, etc)
    - Suspend the application
    - Scale up the application
    - Destroy
3. Lets Upgrade to Data Persistence
  - Install REX-Ray
    - What is REX-Ray?
    - More Info: https://github.com/emccode/rexray
  - Install DVDCLI (Docker Volume Driver CLI)
    - What is DVDCLI?
    - More Info: https://github.com/emccode/dvdcli
  - Install mesos-module-dvdi
    - What is mesos-module-dvdi?
    - More Info: https://github.com/emccode/mesos-module-dvdi
  - Configure the components to interact with AWS.
4. Persistent Storage
  - What did we just do?
  - Create an application with an external volume (AWS EBS)
  - We now have persistent state!
  - Simulate a node failure
  - What other applications does this enable?
5. Hands On Lab: Let's explore Mesos with Persistence
  - Standalone MariaDB (or PostgreSQL) with External Storage
    - Standup MariaDB (or PostgreSQL)
    - Deep Dive: Understanding the External Persistent Volume
    - Simulate a Node failure
    - "Destroy" the Application
  - Restart MariaDB (or PostgreSQL) using REST API
    - Where did the data come from? (Not lost!)
    - Examine the External Mounts via the REST API
    - "Destroy" the Application
6. Hands On Lab: Advanced Example
  - Cluster Couchbase (or PostgreSQL) with External Storage
    - Standup Couchbase (or PostgreSQL)
    - Simulate a Node failure
    - "Destroy" the Application
  - Restart Couchbase (or PostgreSQL) using REST API
    - Where did the data come from? (Not lost!)
    - Examine the External Mounts via the REST API
    - "Destroy" the Application

**Additional Tags:**  
Docker, Mesos, Apache, Apache Mesos, Storage, Persistence, AWS, Development, DevOps, Enterprise

**What’s the takeaway for the audience:**  
Adding external persistent storage to your Apache Mesos cluster will fundamentally change the way in which you use Mesos.

**Audience Level:**  
Advanced

**Prerequisite:**  
Moderate Familiarity with Docker, Moderate Familiarity with Linux Mounts, External Storage Knowledge Preferred

**Conceptual or How-To:**  
How-to

**Tutorial hardware and/or Installation Requirements:**  
None, I will be providing the infrastructure via AWS.

**Video URL:**  
NA

**O’Reill Author:**  
NA

**Recommend or encourage you to submit a proposal:**  
Clinton Kitson <clinton.kitson@emc.com>

**Diversity:**  
No

**Travel & Other Expense:**  
No

**If so, please describe:**  
NA

**Additional Notes:**  
NA
