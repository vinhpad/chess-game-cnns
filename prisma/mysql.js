import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()
const User = prisma.users
const Game = prisma.games
module.exports = {
    prisma,
    User,
    Game
}