写操作MongoDB比传统数据库快的根本原因是Mongo使用的内存映射技术 － 写入数据时候只要在内存里完成就可以返回给应用程序，这样并发量自然就很高。而保存到硬体的操作则在后台异步完成。注意MongoDB在2.4就已经是默认安全写了（具体实现在驱动程序里），所以楼上有同学的回答说是”默认不安全“应该是基于2.2或之前版本的。

读操作MongoDB快的原因是： 1）MongoDB的设计要求你常用的数据（working set)可以在内存里装下。这样大部分操作只需要读内存，自然很快。 2）文档性模式设计一般会是的你所需要的数据都相对集中在一起（内存或硬盘），大家知道硬盘读写耗时最多是随机读写所产生的磁头定位时间，数据集中在一起则减少了关系性数据库需要从各个地方去把数据找过来（然后Join）所耗费的随机读时间

另外一个就是如@王子亭所提到的Mongo是分布式集群所以可以平行扩展。目前一般的百万次并发量都是通过几十上百个节点的集群同时实现。这一点MySQL基本无法做到（或者要花很大定制的代价）



[mongo为什么比mysql快 - SegmentFault 思否](https://segmentfault.com/q/1010000000616467/a-1020000000617108)