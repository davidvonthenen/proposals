**Are you both the submitter and main presenter?**  
Yes

**Session Title:**  
Throwaway Kubernetes Clusters and How to Actually Get There

**Session Format:**  
Solo Presentation

**Level of Expertise for Intended Audience:**  
Intermediate

**Which Cloud Native Computing Foundation (CNCF) hosted software will your presentation be focused on?* Check all that apply.**  
CSI, Cluster API, Velero

**Topic**  
Operations

**Session Description:**  
We have been talking about treating Kubernetes clusters as disposable for some time now. There are multiple efforts within the community to make that dream a reality with projects like Cluster API and kubeadm.

The real problem isn't walking away from a cluster (that's easy), but rather it's realizing what to do about the hundreds or even thousands of applications running within that cluster. That's the truly difficult aspect of treating clusters as throwaway.

What makes those applications indispensable?
- application data (databases, metrics, etc)
- configuration (certificates, accessibility, etc)
- transaction logs
- audit logs for regulatory compliance

In this session, we will cover:
- migrating applications and their data between clusters
- discuss embedding these methodologies into existing community efforts
- enhancing projects like ClusterAPI to leverage these concepts

**Benefits to the Ecosystem:**  
Treating Kubernetes clusters like cattle has been the talk of the town for quite some time, but the reality is we are still very far away from that dream. This talk will discuss going past the headlines and taglines of treating clusters like cattle by examining all the various things needed and produced by applications that make swapping out the clusters underneath them so difficult.

What do to about...
- application data and logs
- stateful applications using persistent volumes
- how monolithic applications complicate migration
- certificate management

This talk will focus on a top-down view of treating clusters as disposal. As users we don't care about the infrastructure underneath, we care about our applications and things required for them to run without problems.


**Primary Speaker Name:**  
David vonThenen

**Primary Speaker Company:**  
VMware

**Primary Speaker Job Title:**  
Cloud Native Engineer

**GitHub Handle:**  
dvonthenen

**Twitter Handle:**  
dvonthenen

**Primary Speaker Biography (provide a biography that includes your employer (if any), ongoing projects and your previous speaking experience).**  
David vonThenen is a Cloud Native Engineer at VMware working in the container orchestrator space specifically around the Kubernetes and CNCF ecosystems. His contributions have spanned a wide variety of projects include Jaeger, Helm, Open Tracing, Prometheus, and Cloud Providers just to name a few. Prior to joining VMware, David was a technical architect and development lead for backup/recovery solutions with a heavy focus on the virtualization space.

**Speaker 1 | Has Speaker 1 spoken at any KubeCon + CloudNativeCon conferences before?**  
Yes

**Speaker 1 | Please provide more details:**  
KubeCon Europe 2019
Intro: Kubernetes VMware SIG - David vonThenen & Steven Wong, VMware

Sched link:
https://kccnceu19.sched.com/event/MPi1/intro-kubernetes-vmware-sig-david-vonthenen-steven-wong-vmware

Recording:
https://www.youtube.com/watch?v=6Uh0jAPEB88&t=1s

**I confirm that none of the speakers above violate the following policy: an individual may only be listed as a speaker on up to two proposals, no matter the session format (Presentation - Solo or Dual, Lightning Talk, Tutorial, or Panel).**  
Yes

**What gender does Speaker 1 identify with?**  
Prefer Not to Answer

**Code of Conduct**  
I Agree

**Final Submission**  
I Agree

**PLEASE READ BEFORE PROCEEDING**  
I have read the above instructions on how to submit this proposal to ensure it makes it to the Review Stage.
