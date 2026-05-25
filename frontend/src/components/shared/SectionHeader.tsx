import { cn } from '@/lib/utils'
import { ReactNode } from 'react'

interface Props {
  title: string
  children?: ReactNode
  className?: string
}

export function SectionHeader({ title, children, className }: Props) {
  return (
    <div className={cn('flex items-center justify-between px-3 py-2 border-b border-zinc-800', className)}>
      <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">{title}</span>
      {children && <div className="flex items-center gap-1">{children}</div>}
    </div>
  )
}
