# 事务的四个条件

原子性，一致性，隔离性，持久化

# 事务的四种类型

扁平事务(Flat Transactions)

带有保存点的扁平事务(Flat Transactions with Savepoints)

链事务(Chained Transactions)

嵌套事务(Nested Transactions)

分布式事务(Distributed Transactions)

**扁平事务**（Flat Transaction）是事务类型中最简单的一种，但在实际生产环境中，这可能是使用最为频繁的事务。在扁平事务中，所有操作都处于同一层次，其由BEGIN WORK开始，由COMMITWORK或ROLLBACK WORK结束，其间的操作是原子的，要么都执行，要么都回滚。

**带有保存点的扁平事务**（Flat Transactions with Savepoint），除了支持扁平事务支持的操作外，允许在事务执行过程中回滚到同一事务中较早的一个状态。这是因为某些事务可能在执行过程中出现的错误并不会导致所有的操作都无效，放弃整个事务不合乎要求，开销也太大。保存点（Savepoint）用来通知系统应该记住事务当前的状态，以便当之后发生错误时，事务能回到保存点当时的状态。

> **保存点用SAVE WORK函数来建立，通知系统记录当前的处理状态。**当出现问题时，保存点能用作内部的重启动点，根据应用逻辑，决定是回到最近一个保存点还是其他更早的保存点。

**链事务**（Chained Transaction）可视为保存点模式的一种变种。带有保存点的扁平事务，当发生系统崩溃时，所有的保存点都将消失，因为其保存点是易失的（volatile），而非持久的（persistent）。

> 链事务的思想是：在提交一个事务时，释放不需要的数据对象，将必要的处理上下文隐式地传给下一个要开始的事务。注意，提交事务操作和开始下一个事务操作将合并为一个原子操作。这意味着下一个事务将看到上一个事务的结果，就好像在一个事务中进行的一样。
>
> 链事务与带有保存点的扁平事务不同的是，带有保存点的扁平事务能回滚到任意正确的保存点。而链事务中的回滚仅限于当前事务，即只能恢复到最近一个的保存点。对于锁的处理，两者也不相同。链事务在执行COMMIT后即释放了当前事务所持有的锁，而带有保存点的扁平事务不影响迄今为止所持有的锁。

**嵌套事务**（Nested Transaction）是一个层次结构框架。由一个顶层事务（top-level transaction）控制着各个层次的事务。顶层事务之下嵌套的事务被称为子事务（subtransaction），其控制每一个局部的变换。

**分布式事务**（Distributed Transactions）通常是一个在分布式环境下运行的扁平事务，因此需要根据数据所在位置访问网络中的不同节点。

对于InnoDB存储引擎来说，其支持扁平事务、带有保存点的事务、链事务、分布式事务。对于嵌套事务，其并不原生支持。

# MySQL事务的四种隔离级别

![image-20210307084015226](images/image-20210307084015226.png)

## READ UNCOMMITTED（读未提交）

该隔离级别的事务会读到其它未提交事务的数据，此现象也称之为 **脏读** 。

那么怎么设置我们的数据库隔离级别为这个呢

1.准备两个终端，在此命名为 mysql 终端 1 和 mysql 终端 2，再准备一张测试表 `test` ，写入一条测试数据并调整隔离级别为 `READ UNCOMMITTED` ，任意一个终端执行即可。

```sql
SET @@session.transaction_isolation = 'READ-UNCOMMITTED';
create database test;
use test;
create table test(id int primary key);
insert into test(id) values(1);
```

2.登录 mysql 终端 1，开启一个事务，将 ID 为 `1` 的记录更新为 `2` 。

```sql
begin;
update test set id = 2 where id = 1;
select * from test; -- 此时看到一条ID为2的记录
```

登录 mysql 终端 2，开启一个事务后查看表中的数据。

```sql
use test;
begin;
select * from test; -- 此时看到一条 ID 为 2 的记录
```

最后一步读取到了 mysql 终端 1 中未提交的事务（没有 commit 提交动作），即产生了 **脏读** ，大部分业务场景都不允许脏读出现，但是此隔离级别下数据库的并发是最好的。

## READ COMMITTED（读提交）

一个事务可以读取另一个已提交的事务，多次读取会造成不一样的结果，此现象称为不可重复读问题，Oracle 和 SQL Server 的默认隔离级别。

1.准备两个终端，在此命名为 mysql 终端 1 和 mysql 终端 2，再准备一张测试表 `test` ，写入一条测试数据并调整隔离级别为 `READ COMMITTED` ，任意一个终端执行即可。

```sql
SET @@session.transaction_isolation = 'READ-COMMITTED';
create database test;
use test;
create table test(id int primary key);
insert into test(id) values(1);
```

2.登录 mysql 终端 1，开启一个事务，将 ID 为 `1` 的记录更新为 `2` ，并确认记录数变更过来。

```sql
begin;
update test set id = 2 where id = 1;
select * from test; -- 此时看到一条记录为 2
```

3.登录 mysql 终端 2，开启一个事务后，查看表中的数据。

```sql
use test;
begin;
select * from test; -- 此时看一条 ID 为 1 的记录
```

4.登录 mysql 终端 1，提交事务。

```sql
commit;
```

5.切换到 mysql 终端 2。

```sql
select * from test; -- 此时看到一条 ID 为 2 的记录
```

mysql 终端 2 在开启了一个事务之后，在第一次读取 `test` 表（此时 mysql 终端 1 的事务还未提交）时 ID 为 `1` ，在第二次读取 `test` 表（此时 mysql 终端 1 的事务已经提交）时 ID 已经变为 `2` ，说明在此隔离级别下已经读取到已提交的事务。

## REPEATABLE READ（可重复读）

该隔离级别是 MySQL 默认的隔离级别，在同一个事务里， `select` 的结果是事务开始时时间点的状态，因此，同样的 `select` 操作读到的结果会是一致的，但是，会有 **幻读** 现象。MySQL 的 InnoDB 引擎可以通过 `next-key locks` 机制（参考下文 [行锁的算法](https://developer.ibm.com/zh/technologies/databases/articles/os-mysql-transaction-isolation-levels-and-locks/#行锁的算法) 一节）来避免幻读。

1.准备两个终端，在此命名为 mysql 终端 1 和 mysql 终端 2，准备一张测试表 test 并调整隔离级别为 `REPEATABLE READ` ，任意一个终端执行即可。

```sql
SET @@session.transaction_isolation = 'REPEATABLE-READ';
create database test;
use test;
create table test(id int primary key,name varchar(20));
```

2.登录 mysql 终端 1，开启一个事务。

```sql
begin;
select * from test; -- 无记录
```

3.登录 mysql 终端 2，开启一个事务。

```sql
begin;
select * from test; -- 无记录
```

4.切换到 mysql 终端 1，增加一条记录并提交。

```sql
insert into test(id,name) values(1,'a');
commit;
```

5.切换到 msyql 终端 2。

```sql
select * from test; --此时查询还是无记录
```

通过这一步可以证明，在该隔离级别下已经读取不到别的已提交的事务，如果想看到 mysql 终端 1 提交的事务，在 mysql 终端 2 将当前事务提交后再次查询就可以读取到 mysql 终端 1 提交的事务。我们接着实验，看看在该隔离级别下是否会存在别的问题。

6.此时接着在 mysql 终端 2 插入一条数据。

```sql
insert into test(id,name) values(1,'b'); -- 此时报主键冲突的错误
```

也许到这里您心里可能会有疑问，明明在第 5 步没有数据，为什么在这里会报错呢？其实这就是该隔离级别下可能产生的问题，MySQL 称之为 **幻读** 。注意我在这里强调的是 MySQL 数据库，Oracle 数据库对于幻读的定义可能有所不同。

## SERIALIZABLE（序列化）

在该隔离级别下事务都是串行顺序执行的，MySQL 数据库的 InnoDB 引擎会给读操作隐式加一把读共享锁，从而避免了脏读、不可重读复读和幻读问题。

1.准备两个终端，在此命名为 mysql 终端 1 和 mysql 终端 2，分别登入 mysql，准备一张测试表 test 并调整隔离级别为 `SERIALIZABLE` ，任意一个终端执行即可。

```sql
SET @@session.transaction_isolation = 'SERIALIZABLE';
create database test;
use test;
create table test(id int primary key);
```

2.登录 mysql 终端 1，开启一个事务，并写入一条数据。

```sql
begin;
insert into test(id) values(1);
```

3.登录 mysql 终端 2，开启一个事务。

```sql
 begin;
select * from test; -- 此时会一直卡住
```

4.立马切换到 mysql 终端 1,提交事务。

```sql
commit;
```

一旦事务提交，msyql 终端 2 会立马返回 ID 为 1 的记录，否则会一直卡住，直到超时，其中超时参数是由 `innodb_lock_wait_timeout` 控制。由于每条 `select` 语句都会加锁，所以该隔离级别的数据库并发能力最弱，但是有些资料表明该结论也不一定对，如果感兴趣，您可以自行做个压力测试。

# 事务如何实现

事务隔离性由锁来实现，原子性、持久性通过数据库的redo log和undo log来完成。redo log称为重做日志，用来保证事务的原子性和持久性。undo log称为回滚日志，用来帮助事务回滚及MVCC的功能。

有的DBA或许会认为undo是redo的逆过程，其实不然。**redo和undo的作用都可以视为是一种恢复操作，redo恢复提交事务修改的页操作，而undo回滚行记录到某个特定版本。**因此两者记录的内容不同，**redo通常是物理日志，记录的是页的物理修改操作。undo是逻辑日志，根据每行记录进行记录**。

#### redo

InnoDB是事务的存储引擎，其通过Force Log at Commit机制实现事务的持久性，即当事务提交（COMMIT）时，必须先将该事务的所有日志写入到重做日志文件进行持久化，待事务的COMMIT操作完成才算完成。在InnoDB存储引擎中，有关事务的日志由两部分组成，即redo log和undo log。**redo log基本上都是顺序写的，在数据库运行时不需要对redo log的文件进行读取操作。而undo log是需要进行随机读写的。**

为了确保每次日志都写入重做日志文件，**在每次将重做日志缓冲写入重做日志文件后，InnoDB存储引擎都需要调用一次fsync操作**。由于fsync的效率取决于磁盘的性能，因此磁盘的性能决定了事务提交的性能，也就是数据库的性能。

**参数innodb_flush_log_at_trx_commit用来控制重做日志刷新到磁盘的策略。该参数的默认值为1，表示事务提交时必须调用一次fsync操作。还可以设置该参数的值为0和2。0表示事务提交时不进行写入重做日志操作，这个操作仅在master thread中完成，而在master thread中每1秒会进行一次重做日志文件的fsync操作。2表示事务提交时将重做日志写入重做日志文件，但仅写入文件系统的缓存中，不进行fsync操作。**在这个设置下，当MySQL数据库发生宕机而操作系统不发生宕机时，并不会导致事务的丢失。而当操作系统宕机时，重启数据库后会丢失未从文件系统缓存刷新到重做日志文件那部分事务。

虽然用户可以通过设置参数innodb_flush_log_at_trx_commit为0或2来提高事务提交的性能，但是需要牢记的是，这种设置方法丧失了事务的ACID特性。

在MySQL数据库中还有一种二进制日志（binlog），其用来进行POINT-IN-TIME（PIT）的恢复及主从复制（Replication）环境的建立。从表面上看其和重做日志非常相似，都是记录了对于数据库操作的日志。然而，从本质上来看，两者有着非常大的不同。

首先，重做日志是在InnoDB存储引擎层产生，而二进制日志是在MySQL数据库的上层产生的，并且二进制日志不仅仅针对于InnoDB存储引擎，MySQL数据库中的任何存储引擎对于数据库的更改都会产生二进制日志。

其次，两种日志记录的内容形式不同。**MySQL数据库上层的二进制日志是一种逻辑日志，其记录的是对应的SQL语句或者对行的逻辑修改。而InnoDB存储引擎层面的重做日志是物理格式日志，其记录的是对于每个页的修改。**

此外，两种日志记录写入磁盘的时间点不同。**二进制日志只在事务提交完成后进行一次写入。而InnoDB存储引擎的重做日志在事务进行中不断地被写入，这表现为日志并不是随事务提交的顺序进行写入的。**

在InnoDB存储引擎中，重做日志都是以512字节进行存储的。这意味着重做日志缓存、重做日志文件都是以块（block）的方式进行保存的，称之为重做日志块（redo log block），每块的大小为512字节。

#### undo

在对数据库进行修改时，InnoDB存储引擎不但会产生redo，还会产生一定量的undo。这样如果用户执行的事务或语句由于某种原因失败了，又或者用户用一条ROLLBACK语句请求回滚，就可以利用这些undo信息将数据回滚到修改之前的样子。

与redo不同，undo存放在数据库内部的一个特殊段（segment）中，这个段称为undo段（undo segment），undo段位于共享表空间内。**undo是逻辑日志，因此只是将数据库逻辑地恢复到原来的样子**。所有修改都被逻辑地取消了，但是数据结构和页本身在回滚之后可能大不相同。

例如，用户执行了一个INSERT 10W条记录的事务，这个事务会导致分配一个新的段，即表空间会增大。在用户执行ROLLBACK时，会将插入的事务进行回滚，但是表空间的大小并不会因此而收缩。因此，当InnoDB存储引擎回滚时，它实际上做的是与先前相反的工作。对于每个INSERT，InnoDB存储引擎会完成一个DELETE；对于每个DELETE，InnoDB存储引擎会执行一个INSERT；对于每个UPDATE，InnoDB存储引擎会执行一个相反的UPDATE，将修改前的行放回去。

除了回滚操作，undo的另一个作用是MVCC，即**在InnoDB存储引擎中MVCC的实现是通过undo来完成。当用户读取一行记录时，若该记录已经被其他事务占用，当前事务可以通过undo读取之前的行版本信息**，以此实现非锁定读取。

最后也是最为重要的一点是，**undo log会产生redo log，也就是undo log的产生会伴随着redo log的产生，这是因为undo log也需要持久性的保护**。

> 参考

[MySQL 事务隔离级别和锁 – IBM Developer](https://developer.ibm.com/zh/technologies/databases/articles/os-mysql-transaction-isolation-levels-and-locks/)

[【MySQL—原理】事务 - SegmentFault 思否](https://segmentfault.com/a/1190000038919732)

