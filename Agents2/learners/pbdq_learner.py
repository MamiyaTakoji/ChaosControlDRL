import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace

class PBDQ_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(PBDQ_Learner, self).__init__(config, policy)
        conactor_optimizer = torch.optim.Adam(self.policy.conactor.parameters(), self.config.learning_rate)
        qnetwork_optimizer = torch.optim.Adam(
        list(self.policy.eval_BDQhead.parameters())
        +list(self.policy.representation.parameters()), 
         self.config.learning_rate)
        self.optimizer = [conactor_optimizer, qnetwork_optimizer]
        conactor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(conactor_optimizer,
                                                                  start_factor=1.0,
                                                                  end_factor=self.end_factor_lr_decay,
                                                                  total_iters=self.config.running_steps)
        qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(qnetwork_optimizer,
                                                                  start_factor=1.0,
                                                                  end_factor=self.end_factor_lr_decay,
                                                                  total_iters=self.config.running_steps)
        self.scheduler = [conactor_lr_scheduler, qnetwork_lr_scheduler]
        self.tau = config.tau
        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()
        self.config = config
    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        hyact_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        disact_batch = hyact_batch[:, 0:self.config.disc_action_dim].long()
        conact_batch = hyact_batch[:, self.config.disc_action_dim:]
        
        # optimize Q-network
        target_conact = self.policy.Atarget(next_batch)
        _, evalAs1, evalQs1 = self.policy.Qeval(obs_batch,conact_batch)
        _, evalAs2, evalQs2 = self.policy.Qeval(next_batch,target_conact)
        num_action_streams = len(evalQs1)
        with torch.no_grad():
            _, targetAs, targetQs = self.policy.Qtarget(next_batch,target_conact)
            targetQs_ = []
            rows = torch.arange(targetQs[0].size(0), device=targetQs[0].device)
            #计算目标Q值
            for i in range(num_action_streams):
                #每一个分量选取evalAs的分量作为Q值然后求和
                targetQ = targetQs[i][rows,evalAs2[i]]
                targetQs_.append(targetQ)
            targetQ_ = sum(targetQs_)/num_action_streams
            targetQ_ = rew_batch + self.gamma*(1 - ter_batch)*targetQ_
            #获取实际（？）Q值
            targetQs_ = [targetQ_ ]*num_action_streams
        preditQs = []
        for i in range(num_action_streams):
            preditQ = evalQs1[i][rows,disact_batch[:,i].to(torch.int32)]
            preditQs.append(preditQ)
        loss = 0
        for i in range(num_action_streams):
            loss += self.mse_loss(preditQs[i],targetQs_[i])
        q_loss = loss/num_action_streams
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()
    
        # optimize actor network
        policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = - policy_q.mean()
        self.optimizer[0].zero_grad()
        p_loss.backward()
        self.optimizer[0].step()
        
        if self.scheduler is not None:
            self.scheduler[0].step()
            self.scheduler[1].step()

        self.policy.soft_update(self.tau)

        if self.distributed_training:
            info = {
                f"Q_loss/rank_{self.rank}": q_loss.item(),
                f"P_loss/rank_{self.rank}": p_loss.item()
            }
        else:
            info = {
                "Q_loss": q_loss.item(),
                "P_loss": p_loss.item()
            }
        
        return info
    # def update(self, **samples):
    #     self.iterations += 1
    #     obs_batch = torch.as_tensor(samples['obs'], device=self.device)
    #     hyact_batch = torch.as_tensor(samples['actions'], device=self.device)
    #     next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
    #     rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
    #     ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
    #     disact_batch = hyact_batch[:, 0:self.config.disc_action_dim].long()
    #     conact_batch = hyact_batch[:, self.config.disc_action_dim:]
        
    #     # optimize Q-network
    #     #计算目标Q值
    #     #采用DDQN的方法表示误差，不行再换
    #     #先获取目标动作
    #     target_conact = self.policy.Atarget(next_batch)
    #     _, evalAs1, evalQs1 = self.policy.Qeval(obs_batch,conact_batch)
    #     _, targetAs, targetQs = self.policy.Qtarget(next_batch,target_conact)
    #     _, evalAs2, evalQs2 = self.policy.Qeval(next_batch,target_conact)
    #     num_action_streams = len(evalQs2)
    #     targetQs_ = []
    #     rows = torch.arange(targetQs[0].size(0), device=targetQs[0].device)
    #     #计算目标Q值
    #     for i in range(num_action_streams):
    #         #每一个分量选取evalAs的分量作为Q值然后求和
    #         targetQ = targetQs[i][rows,evalAs2[i]]
    #         targetQs_.append(targetQ)
    #     targetQ_ = sum(targetQs_)/num_action_streams
    #     targetQ_ = rew_batch + self.gamma*(1 - ter_batch)*targetQ_
    #     #获取实际（？）Q值
    #     targetQs_ = [targetQ_ ]*num_action_streams
    #     preditQs = []
    #     for i in range(num_action_streams):
    #         preditQ = evalQs1[i][rows,disact_batch[:,i].to(torch.int32)]
    #         preditQs.append(preditQ)
    #     loss = 0
    #     for i in range(num_action_streams):
    #         loss += self.mse_loss(preditQs[i],targetQs_[i])
    #     q_loss = loss/num_action_streams
    #     self.optimizer[1].zero_grad()
    #     q_loss.backward()
    #     self.optimizer[1].step()
    
    #     # optimize actor network
    #     policy_q = self.policy.Qpolicy(obs_batch)
    #     p_loss = - policy_q.mean()
    #     self.optimizer[0].zero_grad()
    #     p_loss.backward()
    #     self.optimizer[0].step()
        
    #     if self.scheduler is not None:
    #         self.scheduler[0].step()
    #         self.scheduler[1].step()

    #     self.policy.soft_update(self.tau)

    #     if self.distributed_training:
    #         info = {
    #             f"Q_loss/rank_{self.rank}": q_loss.item(),
    #             f"P_loss/rank_{self.rank}": p_loss.item()
    #         }
    #     else:
    #         info = {
    #             "Q_loss": q_loss.item(),
    #             "P_loss": p_loss.item()
    #         }
        
    #     return info





































