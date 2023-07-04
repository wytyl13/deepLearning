/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-07-03 14:09:15
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-07-03 14:09:15
 * @Description: some problem we have found when we running the model used pytorch
 * 
 * RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
 * this error will happen if you have used the incompletly weight file. you can simple to
 * change one completely weight file to handle this problem.
***********************************************************************/